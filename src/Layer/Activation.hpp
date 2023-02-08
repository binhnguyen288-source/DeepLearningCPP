#pragma once
#include "Layer.hpp"
namespace Layer {

    class ReLU final : public Layer {
    public:
        ReLU(Size input_size) : Layer(input_size, input_size) {}
        ~ReLU() = default;
        void forward() override;
        void backprop(Mat const& d_output) override;
        void flush_gradient() override {}
        void dump_weight(std::string const&, std::string const&) override {}
        void load_weight(std::string const&, std::string const&) override {}
    };

    class Softmax final : public Layer {
    public:
        Softmax(Size input_size) : Layer(input_size, input_size) {}
        ~Softmax() = default;
        void forward() override;
        void backprop(Mat const& d_output) override;
        void flush_gradient() override {}
        void dump_weight(std::string const&, std::string const&) override {}
        void load_weight(std::string const&, std::string const&) override {}
    };

    class Sigmoid final : public Layer {
    public:
        Sigmoid(Size input_size) : Layer(input_size, input_size) {}
        ~Sigmoid() = default;
        void forward() override;
        void backprop(Mat const& d_output) override;
        void flush_gradient() override {}
        void dump_weight(std::string const&, std::string const&) override {}
        void load_weight(std::string const&, std::string const&) override {}
    };
}