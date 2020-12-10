#pragma once
#include <complex>
#include <iostream>
#include <valarray>

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "myFFT.cl"

typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;

class FFT {
public:
    // Recursive FFT (in-place, divide-and-conquer)
    // Higher memory requirements and redundancy although more intuitive
    static void fft(CArray& x);
    // Cooley-Tukey FFT(in-place, breath-first, decimation-in-frequency)
    // Better optimized but less intuitive
    static void fftCooleyTurkey(CArray& x);
    // inverse fft(in-place)
    static void ifft(CArray& x);

    // opencv version
    static CArray fft(const cv::Mat& in, cv::Mat& out);
    static void ifft(const cv::Mat& in, cv::Mat& out);

    // init opencl
    static cv::ocl::Program &opencl_load_kernel();
};

cv::ocl::Program &FFT::opencl_load_kernel()
{
    static std::once_flag s_init;
    static cv::ocl::Program s_program;

    std::call_once(s_init, [](){
        cv::ocl::ProgramSource kernel_progsrc(ProcessFFT);
        cv::String buildflags = "";
        cv::String errmsg;
        s_program.create(kernel_progsrc, buildflags, errmsg);
    });

    return s_program;
}

void FFT::fft(CArray &x) {

    const size_t N = x.size();
    if(N <= 1)
        return;

    //divide
    CArray even = x[std::slice(0, N/2, 2)];
    CArray odd = x[std::slice(1, N/2, 2)];

    //conquer
    fft(even);
    fft(odd);

    //combine
    for(size_t k = 0; k < N/2; ++k)
    {
        Complex t = std::polar(1.0, -2*M_PI*k/N) * odd[k];
        x[k] = even[k] + t;
        x[k + N/2] = even[k] - t;
    }
}

std::valarray<std::complex<double>> Mat2VecCArray(const cv::Mat& input)
{
    if(input.data == nullptr)
        return std::valarray<std::complex<double>>();

    const unsigned int N = input.cols*input.rows;
    std::valarray<std::complex<double>> ret(N);
    uchar* ptrI = input.data;
    for(int i = 0; i < N; ++i)
    {
        ret[i] = Complex(*(ptrI+i));
    }
    return ret;
}

void CArray2Mat(CArray& x, cv::Mat& out)
{
    uchar* ptrO = out.data;
    for(int i = 0; i < x.size(); ++i)
    {
        //ret[i] = static_cast<std::complex<double>>(*(ptrI+i));
        *(ptrO+i) = abs(x[i]);
    }
}

CArray FFT::fft(const cv::Mat &in, cv::Mat &out) {
    CArray data = Mat2VecCArray(in);

    if(in.data == nullptr)
        return data;

    const clock_t startFFT = clock();
    fft(data);
    //printf( "fft time = %f", float( clock()  - startFFT ) /  CLOCKS_PER_SEC);
    std::cout << "fft time = " << float( clock()  - startFFT ) /  CLOCKS_PER_SEC << std::endl;
    CArray2Mat(data, out);
    return data;
}

void FFT::fftCooleyTurkey(CArray &x) {
    //DFT
    unsigned int N = x.size(), k = N, n;
    fft(x);
}

void FFT::ifft(CArray &x) {
    const clock_t startIFFT = clock();

    // conjugate the complex numbers
    x = x.apply(std::conj);

    // forward fft
    fft(x);

    // conjugate
    x = x.apply(std::conj);

    // scale the numbers
    x /= x.size();

    //printf("ifft time = %f", float( clock()  - startIFFT ) /  CLOCKS_PER_SEC);
    std::cout << "ifft time = " <<  float( clock()  - startIFFT ) /  CLOCKS_PER_SEC << std::endl;
}

void FFT::ifft(const cv::Mat &in, cv::Mat &out) {
    if(in.data == nullptr)
        return;

    CArray data = Mat2VecCArray(in);
    ifft(data);
    CArray2Mat(data, out);
}