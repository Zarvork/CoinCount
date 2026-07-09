cmake_minimum_required(VERSION 3.16)
project(coincount CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(coincount
    main.cc
    dataset/dataset_handler.cc
    recognition/feature_extraction.cc
    recognition/feature_extraction.hh
)

add_compile_options(-Wall -Wextra -Werror -pedantic -Wold-style-cast)

target_link_libraries(coincount ${OpenCV_LIBS})