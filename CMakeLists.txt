cmake_minimum_required ( VERSION 2.6 FATAL_ERROR)   #对于cmake版本的最低版本的要求
project(transfer_pcdfile)                                        #建立的工程名，例如源代码目录路径的变量名为CH_DIR                 



find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})
add_definitions(${CUDA_DEFINITIONS})


SET(CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30 --compiler-options '-fPIC'")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --maxrregcount=32)


# find_package(CUDNN REQUIRED)
set(OpenCV_DIR /usr/local/include/opencv4/opencv2)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}   )
link_directories(${OpenCV_LIBRARIES_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

add_executable(test1 cuda_conv.cpp)
target_link_libraries(test1 ${CUDA_LIBRARIES} ${CUDNN_LIBRARIES} ${OPENCV_LIBRARIES}  )
