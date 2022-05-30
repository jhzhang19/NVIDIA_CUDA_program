#include <stdio.h>
#include<iostream>
//导入cuda所需的运行库
#include<cuda_runtime.h>

//计算函数 A+B=C
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < numElements)
    {
        // std::cout<<"the index is: "<<(int)i<<std::endl;
        C[i] = A[i] + B[i];
    }
}

int main(void)
{
    //vector A/B/C的元素总数
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    std::cout<<"Vector addition of "<<numElements<<"elements"<<std::endl;

    //CPU端初始化A/B/C三个向量
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    //初始化
    for(int i = 0; i<numElements; i++)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }
    //在GPU中初始化A/B/C三个向量的空间
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    //把数据A/B从cpu内存复制到GPU中
    std::cout<<"Copy the data from the host memory to device memory"<<std::endl;
    //(目标地址,数据地址,数据大小,方法)
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    //执行GPU kernel函数
    int threadsPerBlock = 256;
    //计算grid中的block数量,就是要能够容纳所有的线程
    int blockPerGrid = (numElements+threadsPerBlock-1)/threadsPerBlock;
    vectorAdd<<<blockPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    //将结果拷贝回CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    for(int i=0; i<numElements; i++)
    {
        if(fabs(h_A[i]+h_B[i]-h_C[i])>1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    std::cout<<"Test Pass"<<std::endl;
    return 0;
}