cmake_minimum_required(VERSION 3.5)
project(main)
SET(CUDA_VERSION 10)
FIND_PACKAGE(CUDA ${CUDA_VERSION} REQUIRED)

add_definitions(-std=c++11)
#set(QMAKE_CXXFLAGS "-std=c++11")
include_directories(/usr/local/cuda/include)
include_directories(/home/zhou/tensorrt/include)
link_directories(/opt/cuda/lib64 /usr/include /usr/include/python3.6)
include_directories(/usr/include/gstreamer-1.0 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include /usr/include/glib-2.0 /usr/include/libxml2 /usr/lib/aarch64-linux-gnu/glib-2.0/include/)

find_package( OpenCV REQUIRED)
find_package( dlib REQUIRED)
find_package ( Boost REQUIRED COMPONENTS system filesystem python)

include_directories(${Boost_INCLUDE_DIRS})

AUX_SOURCE_DIRECTORY(./src DIR_SRCS)
cuda_add_executable( main ${DIR_SRCS} src/pnet_rt.cpp src/common.h src/common.cpp src/pnet_rt.h src/rnet_rt.cpp src/rnet_rt.h src/baseEngine.cpp src/baseEngine.h src/onet_rt.h src/onet_rt.cpp src/kernels.h src/kernels.cu src/face_embedder.h src/face_embedder.cpp src/alignment.h src/alignment.cpp src/face_classifier.h src/face_classifier.cpp src/test.h src/test.cpp)


set(CMAKE_CXX_FLAGS  "-lpthread")
set(CMAKE_CXX_FLAGS  "-O3")

message(${Boost_LIBRARIES})

target_link_libraries( main ${Boost_LIBRARIES})
target_link_libraries( main ${OpenCV_LIBS} )
target_link_libraries( main dlib::dlib )
target_link_libraries( main cudart)
target_link_libraries( main nvinfer)
target_link_libraries( main nvparsers)
target_link_libraries( main /home/nvidia/jetson-inference/build/aarch64/lib/libjetson-utils.so)


target_link_libraries( main /usr/lib/aarch64-linux-gnu/libpython3.6m.so)

