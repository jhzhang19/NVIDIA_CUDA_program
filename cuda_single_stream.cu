#include<stdio.h>
#include<iostream>
#include<cuda.h>
#include<cudnn.h>
#include<cuda_runtime.h>
#include<device_functions.h>

using namespace std;

//(A+B)/2=C
#define N (1024*1024)   //向量长度,每个流执行数据大小
#define FULL (N*20) //全部数据的大小

__global__ void kernel(int *a, int *b, int *c){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < N){
        c[idx] = (a[idx] + b[idx]) / 2;
    }
}

int main(){
    //查询设备属性
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if(!prop.deviceOverlap){
        cout << "Device will not support overlap!" << endl;
        return 0;
    }
    else{
        cout<<prop.deviceOverlap<<" yes"<<endl;
    }

    //初始化计时器时间
    cudaEvent_t start, stop;
    float elapsedTime;
    //声明流和Buffer指针
    cudaStream_t stream;
    int *host_a, *host_b, *host_c;
    int *dev_a, *dev_b, *dev_c;
    //创建计时器
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //初始化流
    cudaStreamCreate(&stream);
    //在GPU端申请内存空间
    cudaMalloc((void **)&dev_a, N * sizeof(int));
    cudaMalloc((void **)&dev_b, N * sizeof(int));
    cudaMalloc((void **)&dev_c, N * sizeof(int));
    //在CPU端申请内存空间,要使用锁页内存
    cudaHostAlloc((void **)&host_a, FULL * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_b, FULL * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_c, FULL * sizeof(int), cudaHostAllocDefault);
    //初始化A,B向量
    for (int i = 0; i < FULL;i++){
        host_a[i] = rand();
        host_b[i] = rand();
    }
    //single stream开始计算
    cudaEventRecord(start, 0);
    //每次传输计算长度为N的数据
    for (int i = 0; i < FULL;i+=N){
        //传输数据到device,并进行计算
        cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
        kernel<<<N / 256, 256, 0, stream>>>(dev_a, dev_b, dev_c);
        //将计算结果从GPU传输到CPU
        cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream);
    }
    //最后需要同步流
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Single Time is:" << float(elapsedTime) << " s" << endl;
    //释放内存
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);
    cudaStreamDestroy(stream);

    return 0;
}