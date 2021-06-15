cmake_minimum_required(VERSION 2.6)
project(yolov5)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES -Xcompiler -fPIC")
set(CMAKE_BUILD_TYPE Release)
set(GPU_ARCHS 60)

#option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

find_package(CUDA REQUIRED)

#opencv
find_package(OpenCV)

include_directories(/usr/local/cuda/include # cuda
		    /usr/include/aarch64-linux-gnu #tensorrt
		    ${OpenCV_INCLUDE_DIRS} # opencv
		    ${PROJECT_SOURCE_DIR}/include
		    )
link_directories(/usr/local/cuda/lib64
	         /usr/lib/aarch64-linux-gnu
		 ${OpenCV_LIBRARY_DIRS})

# tensorrt
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB CPP_SRC src/*.cpp)
file(GLOB CU_SRC src/*.cu)


cuda_add_library(myplugins SHARED ${CU_SRC})
target_link_libraries(myplugins nvinfer cudart)


add_executable(yolov5 ${CPP_SRC})
target_link_libraries(yolov5 myplugins ${OpenCV_LIBS})

add_definitions(-pthread)

