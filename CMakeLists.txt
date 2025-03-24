cmake_minimum_required(VERSION 3.13)
set (CMAKE_CXX_STANDARD 17)

project(Matrix)

add_executable(mat mtest.cc)
target_include_directories(mat PRIVATE include)