cmake_minimum_required(VERSION 3.16)
project(coin_detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(coincount
    main.cc
    dataset/dataset_handler.cc
    recognition/feature_extraction.cc
    recognition/feature_extraction.hh
    recognition/build_training_set.cc
    recognition/build_training_set.hh
    utils/utils.cc
    detection/coin_detection.cc
    benchmark/benchmark.cc
)

add_compile_options(-Wall -Wextra -Werror -pedantic -Wold-style-cast -O2)

target_link_libraries(coincount ${OpenCV_LIBS})