cmake_minimum_required(VERSION 3.12)
project(main LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(main main.cpp)

find_package(yaml-cpp REQUIRED)
find_package(OpenMP)

target_link_libraries(main PRIVATE yaml-cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)
endif()
