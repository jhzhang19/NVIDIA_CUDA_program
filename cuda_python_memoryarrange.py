import numba
from numba import cuda
import math
import numpy as np
import time

#每个block里的thread数量
TPB = 16


@numba.jit(nopython=True)  #使用numba加速cpu处理
def matmul_cpu(A, B, C):
    for y in range(B.shape[1]):
        for x in range(A.shape[0]):
            tmp = 0.
            for k in range(A.shape[1]):
                tmp = A[x,k] * B[k, y]  #矩阵A的第x行与矩阵B的第y列逐元素相乘累加
            C[x, y] = tmp


@cuda.jit
def matmul_gpu(A, B, C):
    row, col = cuda.grid(2) #当前线程在grid中的索引
    if row<C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp
    

@cuda.jit
def matmul_shared_mem(A, B, C):
    #每次利用shared memory 读取一部分数据
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=numba.float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=numba.float32)
    x, y = cuda.grid(2) #当前线程在grid中的block索引

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    if x >=C.shape[0] and y <= C.shape[1]:
        return
    tmp = 0.
    for i in range(int(A.shape[1]/TPB)):
        sA[tx, ty] = A[x, ty+i*TPB] #每次读取矩阵A中TPB长度的一行
        sB[tx, ty] = B[tx+i*TPB, y] #每次读取矩阵B中TPB长度的一列
        cuda.syncthreads()  #此处是同步线程
        for j in range(TPB):
            #计算两个子矩阵相乘
            tmp += sA[tx, j] * sB[j, ty]
            cuda.syncthreads()
        C[x, y] = tmp



#输入数据
A = np.full((TPB*500, TPB*500), 3, np.float)
B = np.full((TPB*500, TPB*500), 4, np.float)
#输出结果 A*B=C
C_cpu = np.full((A.shape[0], B.shape[1]), 0, np.float)

#CPU 处理计时
print("Start processing in CPU")
start_cpu = time.time()
matmul_cpu(A, B, C_cpu)
end_cpu = time.time()
time_cpu = end_cpu - start_cpu
print("CPU process time is: "+ str(time_cpu)+" s")

#GPU处理
#数据传输到gpu上
A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)
C_global_mem = cuda.device_array((A.shape[0], B.shape[1]))
C_shared_mem = cuda.device_array((A.shape[0], B.shape[1])) 

threadsperblock = (TPB, TPB)
blockspergrid_x = int(math.ceil(A.shape[0]/threadsperblock[0]))
blockspergrid_y = int(math.ceil(B.shape[1]/threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

# gpu_global_memory处理计时
print("GPU processing")
start_gpu = time.time()
matmul_gpu[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
cuda.synchronize()
end_gpu = time.time()
time_gpu = end_gpu - start_gpu
C_global_gpu = C_global_mem.copy_to_host()  #传回host
print("GPU time is: "+str(time_gpu)+" s")


#gpu_shared_memory处理计时
start_gpu_shared = time.time()
matmul_shared_mem[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_shared_mem)
cuda.synchronize()
end_gpu_shared = time.time()
time_gpu_shared = end_gpu_shared - start_gpu_shared
print("GPU time(shared memory) is: " + str(time_gpu_shared) + " s")
C_shared_gpu = C_shared_mem.copy_to_host()