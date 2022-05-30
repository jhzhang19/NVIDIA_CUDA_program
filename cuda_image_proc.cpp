#include<cuda.h>
#include<cudnn.h>
#include<cuda_runtime.h>
#include<opencv2/opencv.hpp>
#include<device_functions.h>
#include<iostream>

using namespace std;
using namespace cv;

//cpu实现边沿检测
void sobel_cpu(Mat srcImg, Mat dstImg, int imgHeight, int imgWidth){
    
}

//gpu实现sobel边沿检测
    //3x3卷积核元素定义
        // x0  x1  x2  
        // x3  x4  x5
        // x6  x7  x8
__global__ void sobel_gpu(unsigned char* in, unsigned char* out, int imgHeight, int imgWidth){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    int index = y * imgWidth + x;

    int Gx = 0;
    int Gy = 0;

    unsigned char x0, x1, x2, x3, x4, x5, x6, x7, x8;
    //没有在边缘进行padding,所以没有考虑图像边界处的像素,而且对于边界检测图像边缘一圈的像素
    // 对其影响不大
    if(x>0 && x<imgWidth && y>0 && y<imgHeight){
        x0 = in[(y - 1) * imgWidth + x - 1];//以x4为中心的左上角元素
        x1= in[(y - 1) * imgWidth + x ]; //上方元素
        x2= in[(y - 1) * imgWidth + x + 1 ]; //右上
        x3= in[y * imgWidth + x - 1 ]; //左
        x4= in[y * imgWidth + x ]; //x4
        x5= in[y * imgWidth + x + 1]; //右
        x6= in[(y + 1) * imgWidth + x - 1 ]; //左下
        x7= in[(y + 1) * imgWidth + x ]; //下
        x8= in[(y + 1) * imgWidth + x + 1 ]; //右下

        Gx = x0 + 2 * x3 + x6 - (x2 + 2 * x5 + x8); //x轴边界卷积核卷积操作
        Gy = x6 + 2 * x7 + x8 - (x0 + 2 * x1 + x2); //y轴边界卷积核卷积操作

        out[index] = (abs(Gx) + abs(Gy)) / 2; //输出结果,采用简化算法(|gx|+|gy|)/2
    }
}

int main(){
    //利用opencv接口读取图片
    Mat grayImg = imread("1.jpg", 0);
    int imgWidth = grayImg.cols;
    int imgHeight = grayImg.rows;

    //利用opencv对读取的图片进行去噪处理
    Mat gaussImg;
    GaussianBlur(grayImg, gaussImg, Size(3, 3), 0, 0, BORDER_DEFAULT);

    //cpu结果为dst_cpu,gpu结果为dst_gpu
    Mat dst_cpu(imgHeight, imgWidth, CV_8UC1, Scalar(0));
    Mat dst_gpu(imgHeight, imgWidth, CV_8UC1, Scalar(0));

    //调用sobel_cpu处理图像
    sobel_cpu(gaussImg, dst_cpu, imgHeight, imgWidth);

    //申请指针将它指向gpu空间
    size_t num = imgHeight * imgWidth * sizeof(unsigned char);
    unsigned char *in_gpu;
    unsigned char *out_gpu;
    cudaMalloc((void **)&in_gpu, num);
    cudaMalloc((void **)&out_gpu, num);

    //定义grid和block的维度
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, 
    (imgHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    //将数据从CPU传输到gpu
    cudaMemcpy(in_gpu, gaussImg.data, num, cudaMemcpyHostToDevice);

    //调用在gpu上运行的核函数
    sobel_gpu<<<blocksPerGrid, threadsPerBlock>>>(in_gpu, out_gpu, imgHeight, imgWidth);

    //将计算结果回传到CPU内存
    cudaMemcpy(dst_gpu.data, out_gpu, num, cudaMemcpyDeviceToHost);

    //显示处理结果
    imshow("gpu", dst_gpu);
    imshow("cpu", dst_cpu);
    waitKey(0);

    //释放gpu内存空间
    cudaFree(in_gpu);
    cudaFree(out_gpu);

    return 0;
}