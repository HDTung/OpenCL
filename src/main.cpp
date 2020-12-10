//
// Created by HDTung on 11/19/20.
//
//#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <valarray>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "FFT.hpp"
#include "Process/ProcessWaveletTemplates.hpp"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "pgm.h"
#include <ctime>

#define PI 3.14159265358979

#define MAX_SOURCE_SIZE (0x100000)

#define AMP(a, b) (sqrt((a)*(a)+(b)*(b)))

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue queue = NULL;
cl_program program = NULL;

enum Mode {
    forward = 0,
    inverse = 1
};

int setWorkSize(size_t* gws, size_t* lws, cl_int x, cl_int y)
{
    switch(y) {
        case 1:
            gws[0] = x;
            gws[1] = 1;
            lws[0] = 1;
            lws[1] = 1;
            break;
        default:
            gws[0] = x;
            gws[1] = y;
            lws[0] = 1;
            lws[1] = 1;
            break;
    }

    return 0;
}

int fftCore(cl_mem dst, cl_mem src, cl_mem spin, cl_int m, enum Mode direction)
{
    cl_int ret;

    cl_int iter;
    cl_uint flag;

    cl_int n = 1<<m;

    cl_event kernelDone;

    cl_kernel brev = NULL;
    cl_kernel bfly = NULL;
    cl_kernel norm = NULL;

    brev = clCreateKernel(program, "bitReverse", &ret);
    bfly = clCreateKernel(program, "butterfly", &ret);
    norm = clCreateKernel(program, "norm", &ret);

    size_t gws[2];
    size_t lws[2];

    switch (direction) {
        case Mode::forward:flag = 0x00000000; break;
        case Mode::inverse:flag = 0x80000000; break;
    }

    ret = clSetKernelArg(brev, 0, sizeof(cl_mem), (void *)&dst);
    ret = clSetKernelArg(brev, 1, sizeof(cl_mem), (void *)&src);
    ret = clSetKernelArg(brev, 2, sizeof(cl_int), (void *)&m);
    ret = clSetKernelArg(brev, 3, sizeof(cl_int), (void *)&n);

    ret = clSetKernelArg(bfly, 0, sizeof(cl_mem), (void *)&dst);
    ret = clSetKernelArg(bfly, 1, sizeof(cl_mem), (void *)&spin);
    ret = clSetKernelArg(bfly, 2, sizeof(cl_int), (void *)&m);
    ret = clSetKernelArg(bfly, 3, sizeof(cl_int), (void *)&n);
    ret = clSetKernelArg(bfly, 5, sizeof(cl_uint), (void *)&flag);

    ret = clSetKernelArg(norm, 0, sizeof(cl_mem), (void *)&dst);
    ret = clSetKernelArg(norm, 1, sizeof(cl_int), (void *)&n);

/* Reverse bit ordering */
    setWorkSize(gws, lws, n, n);
    ret = clEnqueueNDRangeKernel(queue, brev, 2, NULL, gws, lws, 0, NULL, NULL);

/* Perform Butterfly Operations*/
    setWorkSize(gws, lws, n/2, n);
    for (iter=1; iter <= m; iter++){
        ret = clSetKernelArg(bfly, 4, sizeof(cl_int), (void *)&iter);
        ret = clEnqueueNDRangeKernel(queue, bfly, 2, NULL, gws, lws, 0, NULL, &kernelDone);
        ret = clWaitForEvents(1, &kernelDone);
    }

    if (direction == inverse) {
        setWorkSize(gws, lws, n, n);
        ret = clEnqueueNDRangeKernel(queue, norm, 2, NULL, gws, lws, 0, NULL, &kernelDone);
        ret = clWaitForEvents(1, &kernelDone);
    }

    ret = clReleaseKernel(bfly);
    ret = clReleaseKernel(brev);
    ret = clReleaseKernel(norm);

    return 0;
}

int fft()
{
    cl_mem xmobj = NULL;
    cl_mem rmobj = NULL;
    cl_mem wmobj = NULL;
    cl_kernel sfac = NULL;
    cl_kernel trns = NULL;
    cl_kernel hpfl = NULL;

    cl_platform_id platform_id = NULL;

    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;

    cl_int ret;

    cl_float2 *xm;
    cl_float2 *rm;
    cl_float2 *wm;

    pgm_t ipgm;
    pgm_t opgm;

    FILE *fp;
    const char fileName[] = "../src/FFT.cl";
    size_t source_size;
    char *source_str;
    cl_int i, j;
    cl_int n;
    cl_int m;

    size_t gws[2];
    size_t lws[2];

/* Load kernel source code */
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    fprintf(stderr, "Load kernel successfully.\n");

    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

/* Read image */
    /*readPGM(&ipgm, "lena.pgm");

    n = ipgm.width;
    m = (cl_int)(log((double)n)/log(2.0));

    fprintf(stderr, "n = %d, m = %d\n", n, m);

    xm = (cl_float2 *)malloc(n * n * sizeof(cl_float2));
    rm = (cl_float2 *)malloc(n * n * sizeof(cl_float2));
    wm = (cl_float2 *)malloc(n / 2 * sizeof(cl_float2));

    for (i=0; i < n; i++) {
        for (j=0; j < n; j++) {
            ((float*)xm)[(2*n*j)+2*i+0] = (float)ipgm.buf[n*j+i];
            ((float*)xm)[(2*n*j)+2*i+1] = (float)0;
        }
    }*/
/* Read image use opencv */
    using namespace cv;
    cv::Mat img_Src = cv::imread("200.png", IMREAD_GRAYSCALE);
    //cv::resize(img_Src, img_Src, cv::Size(img_Src.rows, img_Src.rows), 1, 1, INTER_CUBIC);
    //readPGM(&ipgm, "lena.pgm");

    n = img_Src.cols;
    m = (cl_int)(log((double)n)/log(2.0));

    fprintf(stderr, "n = %d, m = %d\n", n, m);

    xm = (cl_float2 *)malloc(n * n * sizeof(cl_float2));
    rm = (cl_float2 *)malloc(n * n * sizeof(cl_float2));
    wm = (cl_float2 *)malloc(n / 2 * sizeof(cl_float2));
    const clock_t start_copy = clock();
    for (i=0; i < n; i++) {
        for (j=0; j < n; j++) {
            ((float*)xm)[(2*n*j)+2*i+0] = (float)img_Src.data[n*j+i];
            ((float*)xm)[(2*n*j)+2*i+1] = (float)0;
        }
    }
    fprintf( stderr, "copy time = %f\n", float( clock()  - start_copy ) /  CLOCKS_PER_SEC);

/* Get platform/device  */
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

/* Create OpenCL context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

/* Create Command queue */
    queue = clCreateCommandQueue(context, device_id, 0, &ret);

/* Create Buffer Objects */
    xmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float2), NULL, &ret);
    rmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float2), NULL, &ret);
    wmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, (n/2)*sizeof(cl_float2), NULL, &ret);

/* Transfer data to memory buffer */
    ret = clEnqueueWriteBuffer(queue, xmobj, CL_TRUE, 0, n*n*sizeof(cl_float2), xm, 0, NULL, NULL);

/* Create kernel program from source */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

/* Build kernel program */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

/* Create OpenCL Kernel */
    sfac = clCreateKernel(program, "spinFact", &ret);
    trns = clCreateKernel(program, "transpose", &ret);
    hpfl = clCreateKernel(program, "highPassFilter", &ret);

/* Create spin factor */
    ret = clSetKernelArg(sfac, 0, sizeof(cl_mem), (void *)&wmobj);
    ret = clSetKernelArg(sfac, 1, sizeof(cl_int), (void *)&n);
    setWorkSize(gws, lws, n/2, 1);
    ret = clEnqueueNDRangeKernel(queue, sfac, 1, NULL, gws, lws, 0, NULL, NULL);

/* Butterfly Operation */
    fftCore(rmobj, xmobj, wmobj, m, Mode::forward);

/* Transpose matrix */
    ret = clSetKernelArg(trns, 0, sizeof(cl_mem), (void *)&xmobj);
    ret = clSetKernelArg(trns, 1, sizeof(cl_mem), (void *)&rmobj);
    ret = clSetKernelArg(trns, 2, sizeof(cl_int), (void *)&n);
    setWorkSize(gws, lws, n, n);
    ret = clEnqueueNDRangeKernel(queue, trns, 2, NULL, gws, lws, 0, NULL, NULL);

/* Butterfly Operation */
    fftCore(rmobj, xmobj, wmobj, m, Mode::forward);

/* Apply high-pass filter */
    cl_int radius = n/8;
    ret = clSetKernelArg(hpfl, 0, sizeof(cl_mem), (void *)&rmobj);
    ret = clSetKernelArg(hpfl, 1, sizeof(cl_int), (void *)&n);
    ret = clSetKernelArg(hpfl, 2, sizeof(cl_int), (void *)&radius);
    setWorkSize(gws, lws, n, n);
    ret = clEnqueueNDRangeKernel(queue, hpfl, 2, NULL, gws, lws, 0, NULL, NULL);

/* Inverse FFT */

/* Butterfly Operation */
    fftCore(xmobj, rmobj, wmobj, m, Mode::inverse);

/* Transpose matrix */
    ret = clSetKernelArg(trns, 0, sizeof(cl_mem), (void *)&rmobj);
    ret = clSetKernelArg(trns, 1, sizeof(cl_mem), (void *)&xmobj);
    setWorkSize(gws, lws, n, n);
    ret = clEnqueueNDRangeKernel(queue, trns, 2, NULL, gws, lws, 0, NULL, NULL);

/* Butterfly Operation */
    fftCore(xmobj, rmobj, wmobj, m, inverse);

/* Read data from memory buffer */
    ret = clEnqueueReadBuffer(queue, xmobj, CL_TRUE, 0, n*n*sizeof(cl_float2), xm, 0, NULL, NULL);

/*  */
    float* ampd;
    ampd = (float*)malloc(n*n*sizeof(float));
    for (i=0; i < n; i++) {
        for (j=0; j < n; j++) {
            ampd[n*((i))+((j))] = (AMP(((float*)xm)[(2*n*i)+2*j], ((float*)xm)[(2*n*i)+2*j+1]));
        }
    }
    opgm.width = n;
    opgm.height = n;
    normalizeF2PGM(&opgm, ampd);
    free(ampd);

/* Write out image */
    writePGM(&opgm, "output.pgm");

/* Finalizations*/
    ret = clFlush(queue);
    ret = clFinish(queue);
    ret = clReleaseKernel(hpfl);
    ret = clReleaseKernel(trns);
    ret = clReleaseKernel(sfac);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(xmobj);
    ret = clReleaseMemObject(rmobj);
    ret = clReleaseMemObject(wmobj);
    ret = clReleaseCommandQueue(queue);
    ret = clReleaseContext(context);

    destroyPGM(&ipgm);
    destroyPGM(&opgm);

    free(source_str);
    free(wm);
    free(rm);
    free(xm);

    return 0;
}

cv::Mat convertImg(const cv::Mat& img)
{
    //extraxt x and y channels
    cv::Mat xy[2]; //X,Y
    cv::split(img, xy);

    //calculate angle and magnitude
    cv::Mat magnitude, angle;
    cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);

    //translate magnitude to range [0;1]
    double mag_max;
    cv::minMaxLoc(magnitude, 0, &mag_max);
    magnitude.convertTo(magnitude, -1, 1.0 / mag_max);

    //build hsv image
    cv::Mat _hsv[3], hsv;
    _hsv[0] = angle;
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magnitude;
    cv::merge(_hsv, 3, hsv);

    //convert to BGR and show
    cv::Mat bgr;//CV_32FC3 matrix
    hsv.convertTo(bgr, CV_8UC3, 255);
    //cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::cvtColor(bgr, bgr, cv::COLOR_BGR2GRAY);
    //cv::imshow("optical flow", bgr);
    return bgr;
}

int main()
{
    /*int ret =  fft();

    //
    cv::Mat img_Src = cv::imread("200.png", cv::IMREAD_GRAYSCALE);
    //CArray data = Mat2VecComplex(img_Src);
    cv::Mat dst(img_Src.rows, img_Src.cols, CV_8UC1);
    CArray data = FFT::fft(img_Src, dst);
    cv::imwrite("fft.png", dst);

    //reverse
    cv::Mat reverse(img_Src.rows, img_Src.cols, CV_8UC1);
    FFT::ifft(data);
    CArray2Mat(data, reverse);

//    cv::imwrite("reverse.png", reverse);*/
    // test Complex Wavelet Transform

    // Convert input image to complex values
    cv::Mat img_Src = cv::imread("lena.jpg", cv::IMREAD_GRAYSCALE);
    std::cout << "size = " << img_Src.size() << std::endl;
    cv::Mat tmp(img_Src.rows, img_Src.cols, CV_32FC2);
    {
        cv::Mat fimg(img_Src.rows, img_Src.cols, CV_32F);
        cv::Mat zeros(img_Src.rows, img_Src.cols, CV_32F);

        img_Src.convertTo(fimg, CV_32F);
        zeros = 0;

        cv::Mat channels[] = {fimg, zeros};
        cv::merge(channels, 2, tmp);
    }
    //cv::Mat cvt = convertImg(tmp);
    //cv::imwrite("cvt.png", cvt);

    cv::Mat dst(img_Src.rows, img_Src.cols, CV_32FC2);
    const clock_t startWlT = clock();
    ImageProcessing::Wavelet<cv::Mat>::decompose_multilevel(tmp, dst, 2);
    std::cout << "wft time = " << float( clock()  - startWlT ) /  CLOCKS_PER_SEC << std::endl;
    cv::Mat ret = convertImg(dst);
    cv::imwrite("ret.png", ret);
    //reverse
    cv::Mat dst_reverse(img_Src.rows, img_Src.cols, CV_32FC2);
    const clock_t startWlR = clock();
    ImageProcessing::Wavelet<cv::Mat>::compose_multilevel(dst, dst_reverse, 2);
    std::cout << "wfr time = " << float( clock()  - startWlR ) /  CLOCKS_PER_SEC << std::endl;
    cv::Mat retrev = convertImg(dst_reverse);
    cv::imwrite("retrev.png", retrev);

    // test Opencl

    cv::UMat utmp = tmp.getUMat(cv::ACCESS_READ);

    cv::UMat uresult(img_Src.rows, img_Src.cols, CV_32FC2);
    const clock_t startWlTOCL = clock();
    ImageProcessing::Wavelet<cv::UMat>::decompose_multilevel(utmp, uresult, 2);
    std::cout << "wftocl time = " << float( clock()  - startWlTOCL ) /  CLOCKS_PER_SEC << std::endl;
    cv::Mat mret = uresult.getMat(cv::ACCESS_READ);
    cv::Mat mmret = convertImg(mret);
    cv::imwrite("uret.png", mmret);

    //reverse
    cv::UMat uresultrev(img_Src.rows, img_Src.cols, CV_32FC2);
    const clock_t startWlROCL = clock();
    ImageProcessing::Wavelet<cv::UMat>::compose_multilevel(uresult, uresultrev, 2);
    std::cout << "wftocl time = " << float( clock()  - startWlROCL ) /  CLOCKS_PER_SEC << std::endl;
    cv::Mat mrev= uresultrev.getMat(cv::ACCESS_READ);
    cv::Mat mmrev = convertImg(mrev);
    cv::imwrite("urev.png", mmrev);


    return 0;
}