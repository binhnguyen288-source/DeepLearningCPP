#include "Sequential.hpp"



Sequential::Sequential(Size input_size, loss_function choice) : 
    input_size{input_size}, 
    choice{choice} {}


Sequential::~Sequential() {
    for (auto x : layers) {
        delete x;
    }
    layers.clear();
}

Sequential::Sequential(Sequential&& other) : 
    layers{std::move(other.layers)},
    input_size{other.input_size},
    choice{other.choice} {
    other.layers.clear();
    
}

Sequential& Sequential::operator=(Sequential&& other) {
    if (this == &other) [[unlikely]] return *this;
    for (auto x : layers) {
        delete x;
    }
    layers      = std::move(other.layers);
    input_size  = other.input_size;
    choice      = other.choice;
    other.layers.clear();
    return *this;
}

void Sequential::add_dense_layer(int n_units, activation choice) {
    int n_input = layers.empty() ? input_size.total() : layers.back()->get_output_size().total();
    layers.push_back(new Layer::Dense(n_input, n_units, true));
    if (choice == RELU) {
        layers.push_back(new Layer::ReLU(layers.back()->get_output_size()));
    }
    else if (choice == SOFTMAX) {
        layers.push_back(new Layer::Softmax(layers.back()->get_output_size()));
    }
    else if (choice == SIGMOID) {
        layers.push_back(new Layer::Sigmoid(layers.back()->get_output_size()));
    }
}

Layer::Layer* Sequential::add_conv2d_layer(int n_kernel, int k_size, int stride, int padding, activation choice) {
    auto input_sz = layers.empty() ? input_size : layers.back()->get_output_size();
    Layer::Layer* ret = new Layer::Conv2D(input_sz, n_kernel, k_size, stride, padding, true);
    layers.push_back(ret);
    if (choice == RELU) {
        layers.push_back(new Layer::ReLU(layers.back()->get_output_size()));
    }
    else if (choice == SOFTMAX) {
        layers.push_back(new Layer::Softmax(layers.back()->get_output_size()));
    }
    else if (choice == SIGMOID) {
        layers.push_back(new Layer::Sigmoid(layers.back()->get_output_size()));
    }
    return ret;
}

void Sequential::add_maxpooling2d_layer() {
    
    auto input_sz = layers.empty() ? input_size : layers.back()->get_output_size();
    layers.push_back(new Layer::MaxPooling2D(input_sz));
}

void Sequential::add_flatten_layer() {
    
    auto input_sz = layers.empty() ? input_size : layers.back()->get_output_size();
    layers.push_back(new Layer::Flatten(input_sz));
}

void Sequential::add_concat_layer(Layer::Layer* route) {
    layers.push_back(new Layer::UpscalingConcat(layers.back()->get_output_size(), route));
}

void Sequential::add_residual_layer() {
    
    auto input_sz = layers.empty() ? input_size : layers.back()->get_output_size();
    layers.push_back(new Layer::ResidualBlock(input_sz, true));
    layers.push_back(new Layer::ReLU(layers.back()->get_output_size()));
}


void Sequential::dump_weight(std::string const& folder) {
    int idx = 0;
    for (auto x : layers) {
        x->dump_weight(folder, std::to_string(idx++));
    }
}

void Sequential::load_weight(std::string const& folder) {
    int idx = 0;
    for (auto x : layers) {
        x->load_weight(folder, std::to_string(idx++));
    }
}
#include <chrono>
#include <iostream>

Mat Sequential::infer(Mat const& input) const {
    using namespace std::chrono;
    layers[0]->set_input(input);
    layers[0]->forward();
    for (int i = 1; i < layers.size(); ++i) {
        layers[i]->set_input(layers[i - 1]->get_output());
        layers[i]->forward();
    }
    return layers.back()->get_output();
}


void Sequential::backprop(std::vector<Mat>& inputs, std::vector<Mat>& outputs) {
    static constexpr int epochs = 5;
    {
        Scalar err{};
        for (int i = 0; i < outputs.size(); ++i) {
            auto square = [](Scalar x) { return x * x; };
            err += square(infer(inputs[i])[0] - outputs[i][0]);
        }

        std::cout << "Loss: " << err / inputs.size() << std::endl;
    }
    
    constexpr int batch_size = 1;
    for (int _ = epochs; _ > 0; --_) {
        shuffle_data(inputs, outputs);
        for (int batch_base = 0; batch_base + batch_size < inputs.size(); batch_base += batch_size) {
            for (int i = batch_base; i < batch_base + batch_size; ++i) {
                Mat back = infer(inputs[i]);
                backprop_doutput([&]() {
                    if (choice == SUM_SQUARES_ERROR)
                        return back - outputs[i];
                    
                    return 0.0f - outputs[i] / back;
                }());
            }
            
            for (auto layer : layers) {
                layer->flush_gradient();
            }
        }
    }
    Scalar err{};
    for (int i = 0; i < outputs.size(); ++i) {
        auto square = [](Scalar x) { return x * x; };
        err += square(infer(inputs[i])[0] - outputs[i][0]);
    }
    std::cout << "Loss: " << err / inputs.size() << std::endl;
    
}

void Sequential::backprop_test(std::vector<Mat>& inputs, std::vector<Mat>& outputs,
                               std::vector<Mat> const& test_inputs, std::vector<Mat> const& test_outputs) {
    static constexpr int epochs = 3;
    
    constexpr int batch_size = 1;

    for (int _ = epochs; _ > 0; --_) {
        shuffle_data(inputs, outputs);


        for (int batch_base = 0; batch_base + batch_size < inputs.size(); batch_base += batch_size) {
            for (int i = batch_base; i < batch_base + batch_size; ++i) {

                if (i % 1 == 0) {
                    std::cout << i << std::endl;
                    if (i % 1 == 0) {
                        int count{};
                        for (int i = 0; i < test_inputs.size(); ++i) {
                            int predict = infer(test_inputs[i]).argmax();
                            if (test_outputs[i][predict] > 0) {
                                ++count;
                            }
                        }
                        std::cout << "Accuracy: " << count << std::endl;
                    }
                }
                Mat back = infer(inputs[i]);
                backprop_doutput([&]() {
                    if (choice == SUM_SQUARES_ERROR)
                        return back - outputs[i];
                    return 0.0f - outputs[i] / back;
                }());
            }
            
            for (auto layer : layers) {
                layer->flush_gradient();
            }
        }
    }
}
