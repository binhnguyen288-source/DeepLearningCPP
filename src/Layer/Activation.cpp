#include "Activation.hpp"




void Layer::ReLU::forward() {
    Mat const& input = get_input();
    set_output(input.apply([](Scalar x) { 
        return x > 0.0f ? x : 0.1f * x;
    }));
}

void Layer::ReLU::backprop(Mat const& d_output) {
    Mat const& output = get_output();
    accumulate_dinput(
        output.apply([](Scalar x) {
            return x > 0.0f ? 1.0f : 0.1f;
        }) * d_output
    );
}

void Layer::Softmax::forward() {
    Mat const& input = get_input();
    Mat output = input.apply([scale_max = input.max()](Scalar x) {
        return std::exp(x - scale_max);
    });
    Scalar scale = output.sum();
    output /= scale;
    set_output(std::move(output));
}

void Layer::Softmax::backprop(Mat const& d_output) {
    Size const input_size = get_input_size();
    Mat const& output = get_output();
    Mat d_input(1, input_size, true);
    for (int i = 0; i < input_size.total(); ++i) {
        for (int j = 0; j < input_size.total(); ++j) {
            if (i == j) {
                d_input[i] += (output[i] - output[i] * output[i]) * d_output[j];
            }
            else d_input[i] -= output[i] * output[j] * d_output[j];
        } 
    }
    accumulate_dinput(d_input);
}


void Layer::Sigmoid::forward() {
    Mat const& input = get_input();
    set_output(input.apply([](Scalar x) {
        return 1.0f / (1.0f + std::exp(-x));
    }));
}

void Layer::Sigmoid::backprop(Mat const& d_output) {
    Mat const& output = get_output();
    accumulate_dinput(output.apply([](Scalar x) {
        return x - x * x;
    }) * d_output);
}