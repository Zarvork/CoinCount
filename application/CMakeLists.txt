cmake_minimum_required(VERSION 3.16)
project(coin_detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_compile_options(-Wall -Wextra -Werror -pedantic -O2)

add_library(coincount_lib
    dataset/dataset_handler.cc
    recognition/feature_extraction.cc
    recognition/build_training_set.cc
    utils/utils.cc
    detection/coin_detection.cc
    benchmark/benchmark.cc
)
target_link_libraries(coincount_lib PUBLIC ${OpenCV_LIBS})

add_executable(coincount main.cc)
target_link_libraries(coincount PRIVATE coincount_lib)

option(BUILD_TESTING "Build the unit tests" OFF)

if(BUILD_TESTING)
    enable_testing()
    find_package(GTest REQUIRED)

    add_executable(unit_tests
        tests/test_utils.cc
        tests/test_dataset_handler.cc
        tests/test_feature_extraction.cc
        tests/test_build_training_set.cc
    )
    target_link_libraries(unit_tests PRIVATE
        coincount_lib
        GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(unit_tests)
endif()