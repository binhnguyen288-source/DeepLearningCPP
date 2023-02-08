#pragma once
#include "Conv2D.hpp"
#include "Activation.hpp"
namespace Layer {
    class ResidualBlock final : public Layer {
    public:
        ResidualBlock(Size input_size, bool random) : 
        Layer(input_size, input_size),
        w1(input_size, input_size.channels / 2, 1, 1, 0, random),
        w2(Size(input_size.rows, input_size.cols, input_size.channels / 2), input_size.channels, 3, 1, 1, random) {}
        void forward() override;
        void backprop(Mat const& d_output) override;
        void flush_gradient() override;
        void dump_weight(std::string const& folder, std::string const& pre) override;
        void load_weight(std::string const& folder, std::string const& pre) override;

    private:
        Conv2D w1;
        Conv2D w2;
    };
}