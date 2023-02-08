#pragma once
#include "Layer.hpp"

namespace Layer {
    class Flatten final : public Layer {
        public:
        Flatten(Size input_size) : Layer(input_size, Size{input_size.total(), 1, 1}) {}
        void forward() override;
        void backprop(Mat const& d_output) override;
        void flush_gradient() override {}
        void dump_weight(std::string const&, std::string const&) override {}
        void load_weight(std::string const&, std::string const&) override {}
    };
}