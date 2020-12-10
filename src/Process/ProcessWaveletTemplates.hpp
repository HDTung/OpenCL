#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/ocl.hpp>
#include <mutex>
#include "../Accelerator/ProcessWaveletOpenCLKernels.cl"

namespace ImageProcessing
{
    enum DIRECTION
    {
        vertical,
        horizontal
    };
    template <typename M>
    class Wavelet{
    public:
        static void decompose_multilevel(const M& input, M& output, int levelcount);
        static void decompose(const M& input, M& output);
        static void decompose_1d(const M& src, M& dst, DIRECTION direction);

        static void compose_multilevel(const M& input, M& output, int levelcount);
        static void compose(const M& input, M& output);
        static void compose_1d(const M& src, M& dst, DIRECTION direction);

        static cv::ocl::Program &opencl_load_kernel();
    private:
        // Complex Daubechies wavelets
        static constexpr int FILTER_LEN = 6;

        static constexpr float c_lopass[16] = {
                -0.0662912607f, -0.0855816496f,
                0.1104854346f, -0.0855816496f,
                0.6629126074f,  0.1711632992f,
                0.6629126074f,  0.1711632992f,
                0.1104854346f, -0.0855816496f,
                -0.0662912607f, -0.0855816496f,

                0.0f, 0.0f, 0.0f, 0.0f // Padding for float16 access in OpenCL kernel
        };

        static constexpr float c_hipass[16] = {
                -0.0662912607f,  0.0855816496f,
                -0.1104854346f, -0.0855816496f,
                0.6629126074f, -0.1711632992f,
                -0.6629126074f,  0.1711632992f,
                0.1104854346f,  0.0855816496f,
                0.0662912607f, -0.0855816496f,
                0.0f, 0.0f, 0.0f, 0.0f
        };
    };

    template <typename M> constexpr float Wavelet<M>::c_lopass[];
    template <typename M> constexpr float Wavelet<M>::c_hipass[];

    // load kernels function
    template <typename M>
    cv::ocl::Program& Wavelet<M>::opencl_load_kernel() {
        static std::once_flag s_init;
        static cv::ocl::Program s_program;

        std::call_once(s_init, [](){
           cv::ocl::ProgramSource kernel_progsrc(g_wavelet_kernel_src);
           cv::String buildflags = "";
           cv::String errmsg;
           s_program.create(kernel_progsrc, buildflags, errmsg);
        });
        return s_program;
    }


    template <typename M>
    void Wavelet<M>::decompose_multilevel(const M& input, M& output, int levelcount) {
        M tmp(input.rows, input.cols, CV_32FC2);

        for(int i = 0; i < levelcount; ++i)
        {
            int w = input.cols >> i;
            int h = input.rows >> i;

            M srcarea;
            M dstarea = output(cv::Rect(0, 0, w, h));

            if(i == 0)
            {
                srcarea = input(cv::Rect(0, 0, w, h));
            }else
            {
                srcarea = tmp(cv::Rect(0, 0, w, h));
                dstarea.copyTo(srcarea);
            }

            decompose(srcarea, dstarea);
        }
    }

    template <typename M>
    void Wavelet<M>::decompose(const M& input, M& output) {
        // perform by rows and cols
        M tmp(input.rows, input.cols, CV_32FC2);

        decompose_1d(input, tmp, DIRECTION::vertical);
        decompose_1d(tmp, output, DIRECTION::horizontal);
    }

    template <>
    inline void Wavelet<cv::Mat>::decompose_1d(const cv::Mat& src, cv::Mat& dst, DIRECTION direction) {
        int count = direction == DIRECTION::vertical ? src.cols : src.rows;
        int length = direction == DIRECTION::vertical ? src.rows : src.cols;
        int halflen = length /2 ;

        const cv::Vec2f* lopass = reinterpret_cast<const cv::Vec2f*>(c_lopass);
        const cv::Vec2f* hipass = reinterpret_cast<const cv::Vec2f*>(c_hipass);

        for(int x = 0; x < count; ++x)
        {
            for(int y = 0; y < length; y+=2)
            {
                cv::Vec2f lo = 0.0f, hi = 0.0f;
                for(int j = 0; j < FILTER_LEN; ++j)
                {
                    int pos = y + j - FILTER_LEN/2; // y-3 -> y+2
                    if(pos < 0) pos = length + pos;
                    if(pos > length) pos = pos - length;

                    cv::Point xy = direction == DIRECTION::vertical ? cv::Point(x, pos) : cv::Point(pos, x);
                    cv::Vec2f val = src.at<cv::Vec2f>(xy);

                    // multiply 2 complex
                    lo[0] += val[0] * lopass[j][0] - val[1] * lopass[j][1];
                    lo[1] += val[1] * lopass[j][0] + val[0] * lopass[j][1];
                    hi[0] += val[0] * hipass[j][0] - val[1] * hipass[j][1];
                    hi[1] += val[1] * hipass[j][0] + val[0] * hipass[j][1];
                }

                if(direction == DIRECTION::vertical)
                {
                    dst.at<cv::Vec2f>(cv::Point(x, y/2)) = lo;
                    dst.at<cv::Vec2f>(cv::Point(x, y/2 + halflen)) = hi;
                }else
                {
                    dst.at<cv::Vec2f>(cv::Point(y/2, x)) = lo;
                    dst.at<cv::Vec2f>(cv::Point(y/2 + halflen, x)) = hi;
                }
            }
        }
    }

    // 1-dimentional decomposition with OpenCL acceleration
    template <>
    inline void Wavelet<cv::UMat>::decompose_1d(const cv::UMat &src, cv::UMat &dst, DIRECTION direction) {
        cv::ocl::Program &prog = opencl_load_kernel();
        cv::ocl::Kernel kernel;

        size_t globalThreads[2];

        if(direction == DIRECTION::vertical)
        {
            kernel.create("decompose_vertical", prog);
            globalThreads[0] = dst.cols ;
            globalThreads[1] = dst.rows /2 ;
        }else
        {
            kernel.create("decompose_horizontal", prog);
            globalThreads[0] = dst.cols /2 ;
            globalThreads[1] = dst.rows ;
        }

        kernel.args(cv::ocl::KernelArg::ReadOnlyNoSize(src),
                    cv::ocl::KernelArg::WriteOnly(dst),
                    cv::ocl::KernelArg::Constant(c_lopass, sizeof(float) * 16),
                    cv::ocl::KernelArg::Constant(c_hipass, sizeof(float) * 16));
        if(kernel.run(2, globalThreads, NULL, true) == false)
        {
            throw std::runtime_error("Failed to execute OpenCL kernel");
        }
    }

    template <typename M>
    void Wavelet<M>::compose_multilevel(const M &input, M &output, int levelcount) {
        M tmp(input.rows, input.cols, CV_32FC2);

        input.copyTo(tmp);

        for(int i = levelcount - 1; i >= 0; --i)
        {
            int w = input.cols >> i;
            int h = input.rows >> i;
            M srcarea = tmp(cv::Rect(0, 0, w, h));
            M dstarea = output(cv::Rect(0, 0, w, h));

            compose(srcarea, dstarea);

            dstarea.copyTo(tmp(cv::Rect(0, 0, w, h)));
        }
    }

    template <typename M>
    void Wavelet<M>::compose(const M &input, M &output) {
        M tmp(input.rows, input.cols, CV_32FC2);

        compose_1d(input, tmp, DIRECTION::vertical);
        compose_1d(tmp, output, DIRECTION::horizontal);
    }

    template <>
    inline void Wavelet<cv::Mat>::compose_1d(const cv::Mat &src, cv::Mat &dst, DIRECTION direction) {
        int count = direction == DIRECTION::vertical ? src.cols : src.rows;
        int length = direction == DIRECTION::vertical ? src.rows : src.cols;
        int halflen = length /2 ;
        const cv::Vec2f* lopass = reinterpret_cast<const cv::Vec2f*>(c_lopass);
        const cv::Vec2f* hipass = reinterpret_cast<const cv::Vec2f*>(c_hipass);

        for(int x  = 0 ; x < count; ++x)
        {
            for(int y = 0; y < length; y ++)
            {
                cv::Vec2f ret = 0.0f;
                for (int j = (y + FILTER_LEN / 2) % 2; j < FILTER_LEN; j += 2)// 1,3,5 || 0,2,4
                {
                    int pos = (y - j + FILTER_LEN / 2) / 2; // y/2 -1 0 +1 || y/2 -1/2 +1/2 +3/2
                    if(pos < 0) pos = pos + halflen;
                    if(pos >= halflen) pos = pos - halflen;

                    cv::Vec2f val_lo = src.at<cv::Vec2f>(direction == DIRECTION::vertical ? cv::Point(x, pos) : cv::Point(pos, x));
                    cv::Vec2f val_hi = src.at<cv::Vec2f>(direction == DIRECTION::vertical ? cv::Point(x, pos + halflen) : cv::Point(pos + halflen, x));

                    ret[0] += val_lo[0] * lopass[j][0] + val_hi[0] * hipass[j][0];
                    ret[0] += val_lo[1] * lopass[j][1] + val_hi[1] * hipass[j][1];
                    ret[1] += val_lo[1] * lopass[j][0] + val_hi[1] * hipass[j][0];
                    ret[1] -= val_lo[0] * lopass[j][1] + val_hi[0] * hipass[j][1];
                }
                if(direction == DIRECTION::vertical)
                {
                    dst.at<cv::Vec2f>(cv::Point(x,y)) = ret;
                }
                else
                {
                    dst.at<cv::Vec2f>(cv::Point(y,x)) = ret;
                }
            }
        }
    }

    template <>
    inline void Wavelet<cv::UMat>::compose_1d(const cv::UMat &src, cv::UMat &dst, DIRECTION direction) {
        cv::ocl::Program &prog = opencl_load_kernel();
        cv::ocl::Kernel kernel;

        size_t globalThreads[2];

        if(direction == DIRECTION::vertical)
        {
            kernel.create("compose_vertical", prog);
            globalThreads[0] = dst.cols ;
            globalThreads[1] = dst.rows /2 ;
        }else
        {
            kernel.create("compose_horizontal", prog);
            globalThreads[0] = dst.cols /2 ;
            globalThreads[1] = dst.rows ;
        }

        kernel.args(cv::ocl::KernelArg::ReadOnlyNoSize(src),
                    cv::ocl::KernelArg::WriteOnly(dst),
                    cv::ocl::KernelArg::Constant(c_lopass, sizeof(float) * 16),
                    cv::ocl::KernelArg::Constant(c_hipass, sizeof(float) * 16));
        if(kernel.run(2, globalThreads, NULL, true) == false)
        {
            throw std::runtime_error("Failed to execute OpenCL kernel");
        }
    }
}