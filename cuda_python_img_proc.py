import imp
import cv2 as cv
# print(cv.__version__)
import numpy as np
import numba
from numba import cuda
import time
import math

@cuda.jit  #标注为gpu执行
def process_gpu(img, channels):
    #计算线程在全局数据下的索引
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    for c in range (channels):
        color = img[tx, ty][c] * 2.0 + 30.0 #每个通道的像素值都增大
        #对像素范围进行限定
        if color > 255:
            img[tx, ty][c] = 255
        elif color < 0:
            img[tx, ty][c] = 0
        else:
            img[tx, ty][c] = color


def process_cpu(img, dst):
    rows, cols, channels = img.shape
    for i in range(rows):
        for j in range(cols):
            for c in range(channels):
                color = img[i, j][c] * 2.0 + 30.0
                if color > 255:
                    dst[i, j][c] = 255
                elif color < 0:
                    dst[i, j][c] = 0
                else:
                    dst[i, j][c] = color



if __name__ == "__main__":
    #载入图片
    img = cv.imread("test.png")
    #读取图片像素行列信息
    rows, clos, channels = img.shape
    #cpu,gpu处理的数据
    dst_cpu = img.copy()
    dst_gpu = img.copy()

    #调用函数进行处理
    #gpu处理
    dImg = cuda.to_device(img)  #将图片数据拷贝到device上
    #设置线程/block数量
    threadsperblock = (16, 16)  #数量为16倍数,最大不超过显卡限制
    blockspergrid_x = int(math.ceil(rows/threadsperblock[0]))   #往上取整是为了让线程覆盖所有图像像素,防止遗漏像素,block个数是32的倍数
    blockspergrid_y = int(math.ceil(clos/threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    #同步一下cpu和device的计算进度
    cuda.synchronize()
    #gpu处理时间
    print("GPU processing:")
    start_gpu = time.time()
    process_gpu[blockspergrid, threadsperblock](dImg, channels)
    cuda.synchronize()
    end_gpu = time.time()
    time_gpu = end_gpu - start_gpu
    dst_gpu = dImg.copy_to_host()
    print("GPU process time is: " + str(time_gpu) + "s")
    
    #cpu处理
    print("CPU processing:")
    start_cpu = time.time()
    process_cpu(img, dst_cpu)
    end_cpu = time.time()
    time_cpu = end_cpu - start_cpu
    print("CPU process time is: "+ str(time_cpu) + "s")

    #保存处理结果
    cv.imwrite("result_cpu.png", dst_cpu)
    cv.imwrite("result_gpu.png", dst_gpu)
    print("Process Done!")