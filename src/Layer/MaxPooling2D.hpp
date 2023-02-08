#pragma once
#include "Layer.hpp"
namespace Layer {
    class MaxPooling2D final : public Layer {
    public:
        MaxPooling2D(Size input_size) : Layer(
            input_size,
            Size{input_size.rows / 2, input_size.cols / 2, input_size.channels}
        ) {}
        void forward() override;
        void backprop(Mat const& d_output) override;

        void flush_gradient() override {}

        void dump_weight(std::string const&, std::string const&) override {}
        void load_weight(std::string const&, std::string const&) override {}
    };
}