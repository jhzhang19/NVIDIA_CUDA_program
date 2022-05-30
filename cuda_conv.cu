#include"cuda_runtime.h"
#include<cudnn.h>
#include<cuda.h>
#include<device_functions.h>
#include<opencv4/opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;

float3 data_kernel[] = {
    make_float3(-1.0f, -1.0f, -1.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f),
    make_float3(-2.0f, -2.0f, -2.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(2.0f, 2.0f, 2.0f),
    make_float3(-1.0f, -1.0f, -1.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f),
    make_float3(-1.0f, -1.0f, -1.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f),
    make_float3(-2.0f, -2.0f, -2.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(2.0f, 2.0f, 2.0f),
    make_float3(-1.0f, -1.0f, -1.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f),
    make_float3(-1.0f, -1.0f, -1.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f),
    make_float3(-2.0f, -2.0f, -2.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(2.0f, 2.0f, 2.0f),
    make_float3(-1.0f, -1.0f, -1.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f),

};

int main(){
    //利用opencv的接口读取图片相关信息
    cv::Mat img = cv::imread("/home/zjh19/图片/00000.png");
    int imgWidth = img.cols;
    int imgHeight = img.rows;
    int imgChannel = img.channels();

    cv::Mat dst_gpu(imgHeight, imgWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    size_t num = imgChannel * imgHeight * imgWidth * sizeof(unsigned char);

    // 1.在gpu上分配空间
    unsigned char *in_gpu; //输入gpu的图像数据
    unsigned char *out_gpu; //输出gpu的图像数据
    float *filt_data;
    cudaMalloc((void **)&filt_data, 3 * 3 * 3 * sizeof(float3));
    cudaMalloc((void **)&in_gpu, num);
    cudaMalloc((void **)*out_gpu, num);

    // 2.初始化句柄   
    cudnnHandle_t handle;
    cudnnCreate(&handle);
    
    // 3.描述tensor
    //input descriptor
    cudnnTensorDescriptor_t input_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NHWC, 
    CUDNN_DATA_FLOAT, 1, 3, imgHeight, imgWidth);
    //output descriptor
    cudnnTensorDescriptor_t output_descriptor;
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NHWC, 
    CUDNN_DATA_FLOAT, 1, 3, imgHeight, imgWidth);
    //kernel descriptor
    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnCreateFilterDescriptor(&kernel_descriptor);
    cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_FLOAT, 
    CUDNN_TENSOR_NCHW, 3, 3, 3, 3);
    
    // 4.描述操作并设置相关参数
    cudnnConvolutionDescriptor_t conv_descriptor;
    cudnnCreateConvolutionDescriptor(&conv_descriptor);
    cudnnSetConvolution2dDescriptor(conv_descriptor, 1, 1, 1, 1, 1, 1, 
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // 5.描述算法,让计算机自动选择最佳算法
    cudnnConvolutionFwdAlgoPerf_t algo;
    cudnnGetConvolutionForwardAlgorithm_v7(handle, input_descriptor, kernel_descriptor, 
    conv_descriptor, output_descriptor, 1, 0, &algo);


    // 6.申请工作空间
    size_t workspace_size = 0;
    //计算工作空间大小
    cudnnGetConvolutionForwardWorkspaceSize(handle, input_descriptor, kernel_descriptor, conv_descriptor, output_descriptor, algo.algo, &workspace_size);
    //分配工作空间
    void *workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);

    // 7.将计算需要的数据传输到GPU
    cudaMemcpy((void *)filt_data, (void *)data_kernel, 3 * 3 * 3 * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(in_gpu, img.data, num, cudaMemcpyHostToDevice);

    // 8.开始计算
    auto alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(handle, &alpha, input_descriptor, in_gpu, 
    kernel_descriptor, filt_data, conv_descriptor, algo.algo, &workspace, workspace_size, &beta, output_descriptor, out_gpu);

    // 9.将计算结果回传到CPU
    cudaMemcpy(dst_gpu.data, out_gpu, num, cudaMemcpyDeviceToHost);

    // 10.释放资源
    cudaFree(in_gpu);
    cudaFree(out_gpu);
    cudaFree(workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(conv_descriptor);

    cudnnDestroy(handle);

    return 0;
}
