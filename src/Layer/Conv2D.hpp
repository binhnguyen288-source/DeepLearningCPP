#pragma once
#include "Layer.hpp"
namespace Layer {
    class Conv2D final : public Layer {
    public:
        Conv2D(Size size, int n_kernel, int k_size, int stride, int padding, bool random);
        void forward() override;
        void backprop(Mat const& d_output) override;
        void flush_gradient() override;
        void dump_weight(std::string const& folder, std::string const& pre) override;
        void load_weight(std::string const& folder, std::string const& pre) override;
    private:
        Mat kernels;
        Mat dkernels;
        Mat vk;
        int n_kernel, k_size;
        int stride, padding;
    };
}

