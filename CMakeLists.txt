cmake_minimum_required(VERSION 3.2)
project(yolov5)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES -Xcompiler -fPIC")
set(CMAKE_BUILD_TYPE Release)
set(GPU_ARCHS 60)

#option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(CUDA_TOOLKIT_ROOT_DIR /home/lichenguang/Workspace/software/cuda/v9.2)
find_package(CUDA REQUIRED)

#opencv
set(OpenCV_DIR /home/lichenguang/Workspace/software/opencv)

include_directories(${CUDA_INCLUDE_DIRS} # cuda
            /home/lichenguang/Workspace/software/TensorRT-4.0.1.6/include #tensorrt
            ${OpenCV_DIR}/include # opencv
            ${PROJECT_SOURCE_DIR}/include
            )
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64
                 /home/lichenguang/Workspace/software/TensorRT-4.0.1.6/lib
				 ${OpenCV_DIR}/lib
				 ${PROJECT_SOURCE_DIR}/lib)

file(GLOB CPP_SRC src/*.cpp)
file(GLOB CU_SRC src/*.cu)


cuda_add_library(cuda_ctdet SHARED ${CU_SRC})
target_link_libraries(cuda_ctdet nvinfer cudart)


add_executable(rundet ${CPP_SRC} main.cpp)
add_executable(runvideo ${CPP_SRC} run_video.cpp)
add_library(ctdet SHARED ${CPP_SRC})
target_link_libraries(rundet cuda_ctdet opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_videoio)
target_link_libraries(ctdet cuda_ctdet opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_videoio)
target_link_libraries(runvideo cuda_ctdet opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_videoio)

add_definitions(-O2 -pthread)
