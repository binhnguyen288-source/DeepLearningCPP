#pragma once
#include "Layers.hpp"
class Sequential {
    
public:
    enum loss_function {
        CATEGORICAL_CROSSENTROPY,
        SUM_SQUARES_ERROR
    };
    Sequential() = default;
    Sequential(std::vector<Layer::Layer*> const& view, Size input_size) : layers{view}, input_size{input_size} {}
    Sequential(Size input_size, loss_function choice);
    ~Sequential();
    Sequential(Sequential&& other);
    Sequential(Sequential const& other) = delete;
    Sequential& operator=(Sequential&& other);
    Sequential& operator=(Sequential const& other) = delete;
    
    void add_dense_layer(int n_units, activation choice);
    Layer::Layer* add_conv2d_layer(int n_kernel, int k_size, int stride, int padding, activation choice = RELU);
    void add_maxpooling2d_layer();
    void add_flatten_layer();
    void add_residual_layer();
    void add_concat_layer(Layer::Layer* route);
    void dump_weight(std::string const& folder);
    void load_weight(std::string const& folder);
    Mat infer(Mat const& input) const;
    void backprop(std::vector<Mat>& inputs, std::vector<Mat>& outputs);

    void backprop_test(std::vector<Mat>& inputs, std::vector<Mat>& outputs,
                       std::vector<Mat> const& test_inputs, std::vector<Mat> const& test_outputs);
    void backprop_doutput(Mat d_output) {
        for (int L = layers.size() - 1; L >= 0; --L) {
            layers[L]->backprop(d_output);
            d_output = layers[L]->get_dinput();
        }
    }
    void flush_gradients() {
        for (auto l : layers) {
            l->flush_gradient();
        }
    }

    std::vector<Layer::Layer*> get_layer_view() {
        return layers;
    }

    Size get_input_size() {
        return input_size;
    }
    
private:
    std::vector<Layer::Layer*> layers;
    Size input_size{0, 0, 0};
    loss_function choice{SUM_SQUARES_ERROR};
    
};