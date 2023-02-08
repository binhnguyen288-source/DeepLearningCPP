
#pragma once
#include <fstream>
#include <vector>
#include <array>
#include "../Mat/Mat.hpp"
#include <iostream>

uint32_t readu32_little_endian(char* buffer, int idx) {
    uint8_t* ptr = reinterpret_cast<uint8_t*>(buffer) + idx;
    return (ptr[0] << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];
}

std::vector<Mat> read_mnist_images(std::string const& filename) {
    std::vector<Mat> result;
    static char buffer[1 << 26];

    std::ifstream file(filename, std::ifstream::binary);
    file.read(buffer, 1 << 26);
    file.close();

    const uint32_t magic_number = readu32_little_endian(buffer, 0);
    const uint32_t n_samples = readu32_little_endian(buffer, 4);

    for (int i = 0; i < n_samples; ++i) {

        Mat img(1, 28, 28, 1, false);
        
        
        std::transform(buffer + 16 + 784 * i, buffer + 16 + 784 * (i + 1), 
                       img.begin(), [](char c) -> Scalar { return (unsigned char)c / static_cast<Scalar>(255.0); });
        result.push_back(std::move(img));
    }
    return result;
}

std::vector<Mat> read_mnist_labels(std::string const& filename) {
    std::vector<Mat> result;
    static char buffer[1 << 26];

    std::ifstream file(filename, std::ifstream::binary);
    file.read(buffer, 1 << 26);
    file.close();

    const uint32_t magic_number = readu32_little_endian(buffer, 0);
    const uint32_t n_samples = readu32_little_endian(buffer, 4);
    for (int i = 0; i < n_samples; ++i) {
        Mat label(1, 10, 1, 1, true);
        label[buffer[i + 8]] = 1.0;
        result.push_back(std::move(label));
    }

    return result;
}