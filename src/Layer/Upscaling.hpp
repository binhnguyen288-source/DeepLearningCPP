#pragma once
#include "Layer.hpp"

namespace Layer {
    class UpscalingConcat final : public Layer {
    public:
        UpscalingConcat(Size input_size, Layer* route_layer) : Layer(
            input_size, 
            Size{input_size.rows * 2, 
                 input_size.cols * 2, 
                 (route_layer->get_input_size().channels + input_size.channels)}
        ), route_layer{route_layer} {}
        void forward() override;
        void backprop(Mat const& d_output) override;
        void flush_gradient() override {}
        void dump_weight(std::string const&, std::string const&) override {}
        void load_weight(std::string const&, std::string const&) override {}
    private:
        Layer* route_layer;
    };
}