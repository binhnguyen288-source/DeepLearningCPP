cmake_minimum_required(VERSION 3.18)
project(cnn)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD_REQUIRED 20)
add_compile_options(-Ofast -mcpu=apple-m1 -DNDEBUG -std=c++20 -fno-exceptions)
add_link_options(-fno-exceptions -DNDEBUG)


add_library(
    Layer STATIC
    "src/Layer/Conv2D.cpp"
    "src/Layer/Dense.cpp"
    "src/Layer/Flatten.cpp"
    "src/Layer/MaxPooling2D.cpp"
    "src/Layer/Upscaling.cpp"
    "src/Layer/ResidualBlock.cpp"
    "src/Layer/Sequential.cpp"
    "src/Layer/Activation.cpp"
)

add_library(
    Mat STATIC
    "src/Mat/Mat.cpp"
)


add_executable(test src/test.cpp)


target_link_libraries(test Layer Mat)


