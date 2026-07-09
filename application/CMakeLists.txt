cmake_minimum_required(VERSION 3.16)
project(coincount CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(coincount
    main.cc
)

# when they will be created, add files to the executable like 
#add_executable(coincount
#    main.cc
#    recognition/...
#    benchmark/...
#    detection/...
#    dataset/...
#)

target_link_libraries(coincount ${OpenCV_LIBS})