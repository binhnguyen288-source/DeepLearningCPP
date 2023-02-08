#include "Conv2D.hpp"
// #include <thread>
// #include <future>

Layer::Conv2D::Conv2D(Size input_size, int n_kernel, int k_size, int stride, int padding, bool random) :
    Layer(input_size, Size{(input_size.rows - k_size + 2 * padding + 1) / stride, 
                            (input_size.cols - k_size + 2 * padding + 1) / stride, 
                            n_kernel}),
    kernels(k_size, k_size, input_size.channels, n_kernel, false, random),
    dkernels(k_size, k_size, input_size.channels, n_kernel, true),
    vk(k_size, k_size, input_size.channels, n_kernel, true),
    n_kernel{n_kernel}, k_size{k_size},
    stride{stride}, padding{padding} 
{}
#include <thread>
#include <future>
static const int n_workers = std::thread::hardware_concurrency();
void Layer::Conv2D::forward() {

    const Size output_size = get_output_size();
    Mat const& input = get_input();

    auto job_i = [&input, input_size = get_input_size(), &output_size, this](int i_lower, int i_upper) {
        Mat local_output(1, output_size, true);
        for (int k = 0; k < k_size; ++k) {
            int i_input = i_lower * stride + k - padding;
            for (int i = i_lower; i < i_upper; ++i, i_input += stride) {
                if (i_input < 0 || i_input >= input_size.rows) continue;
                for (int t = 0; t < k_size; ++t) {
                    for (int j = 0, j_input = t - padding; j < output_size.cols; ++j, j_input += stride) {
                        if (j_input < 0 || j_input >= input_size.cols) continue;
                        for (int c = 0; c < input_size.channels; ++c) {
                            for (int k_idx = 0; k_idx < n_kernel; ++k_idx) {
                                local_output(i, j, k_idx) += input(i_input, j_input, c) * kernels(k, t, c, k_idx);
                            }
                        }
                    }
                }
            }

        }
        return local_output;
    };

    
    std::vector<std::future<Mat>> futs;
    for (int i = 0; i < n_workers; ++i) {
        int lower = i * output_size.rows / n_workers;
        int upper = (i + 1) * output_size.rows / n_workers;
        futs.push_back(std::async(std::launch::async, job_i, lower, upper));
    }
    Mat output = futs.front().get();
    for (int i = 1; i < n_workers; ++i) {
        output += futs[i].get();
    }

    set_output(std::move(output));
}

void Layer::Conv2D::backprop(Mat const& d_output) {
    Size const output_size = get_output_size();
    Mat const& input = get_input();

     auto job_i = [input_size = get_input_size(), &output_size, this, &d_output, &input](int i_lower, int i_upper) {
        Mat local_dkernel(k_size, k_size, input_size.channels, n_kernel, true);
        Mat local_dinput(1, input_size.rows, input_size.cols, input_size.channels, true);
         
        for (int k = 0; k < k_size; ++k) {
            int i_input = i_lower * stride + k - padding;
            for (int i = i_lower; i < i_upper; ++i, i_input += stride) {
                if (i_input < 0 || i_input >= input_size.rows) continue;
                for (int t = 0; t < k_size; ++t) {
                    for (int j = 0, j_input = t - padding; j < output_size.cols; ++j, j_input += stride) {
                        if (j_input < 0 || j_input >= input_size.cols) continue;
                        for (int c = 0; c < input_size.channels; ++c) {
                            for (int k_idx = 0; k_idx < n_kernel; ++k_idx) {
                                local_dkernel(k, t, c, k_idx) += input(i_input, j_input, c) * d_output(i, j, k_idx);
                            }
                        }
                    }
                }
            }   
        }

        for (int k = 0; k < k_size; ++k) {
            int i_input = i_lower * stride + k - padding;
            for (int i = i_lower; i < i_upper; ++i, i_input += stride) {
                if (i_input < 0 || i_input >= input_size.rows) continue;
                for (int t = 0; t < k_size; ++t) {
                    for (int j = 0, j_input = t - padding; j < output_size.cols; ++j, j_input += stride) {
                        if (j_input < 0 || j_input >= input_size.cols) continue;
                        for (int c = 0; c < input_size.channels; ++c) {
                            for (int k_idx = 0; k_idx < n_kernel; ++k_idx) {
                                local_dinput(i_input, j_input, c) += kernels(k, t, c, k_idx) * d_output(i, j, k_idx);
                            }
                        }
                    }
                }
            }   
        }
        

        return std::make_pair(std::move(local_dkernel), std::move(local_dinput));
    };

    std::vector<std::future<std::pair<Mat, Mat>>> futs;
    for (int i = 0; i < n_workers; ++i) {
        int lower = i * output_size.rows / n_workers;
        int upper = (i + 1) * output_size.rows / n_workers;
        futs.push_back(std::async(std::launch::async, job_i, lower, upper));
    }
    for (int i = 0; i < n_workers; ++i) {
        auto const [worker_dkernel, worker_dinput] = futs[i].get();
        dkernels += worker_dkernel;
        accumulate_dinput(worker_dinput);
    }
}

void Layer::Conv2D::flush_gradient() {
    static const Scalar beta = 0.9f;
    vk = lr * beta * vk + lr * (1.0f - beta) * dkernels;   
    kernels -= vk;
    dkernels.fill_zero();
}

void Layer::Conv2D::dump_weight(std::string const& folder, std::string const& pre) {
    kernels.dump(folder + "/" + pre + "k.bin");
}
void Layer::Conv2D::load_weight(std::string const& folder, std::string const& pre) {
    kernels.load(folder + "/" + pre + "k.bin");
}