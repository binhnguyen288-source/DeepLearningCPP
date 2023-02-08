#pragma once
#include "Layer.hpp"

namespace Layer {
    class Dense final : public Layer {
    public:
        Dense(int n_input, int n_output, bool random);
        void forward() override;
        void backprop(Mat const& d_output) override;
        void flush_gradient() override;
        void dump_weight(std::string const& folder, std::string const& pre) override;
        void load_weight(std::string const& folder, std::string const& pre) override;
    private:
        Mat weight;
        Mat dweight;
        Mat vw;
        Mat bias;
        Mat dbias;
        Mat vb;
    };
}