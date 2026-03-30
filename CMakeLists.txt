cmake_minimum_required(VERSION 3.12)
project(probability_diffusion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(probability_diffusion main.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(probability_diffusion PRIVATE OpenMP::OpenMP_CXX)
endif()
