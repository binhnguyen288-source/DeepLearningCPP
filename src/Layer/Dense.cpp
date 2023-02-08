#include "Dense.hpp"
Layer::Dense::Dense(int n_input, 
                    int n_output, 
                    bool random) : 
    Layer(Size{n_input, 1, 1}, Size{n_output, 1, 1}),
    weight(1, n_output, n_input, 1, false, random),
    dweight(1, n_output, n_input, 1, true),
    vw(1, n_output, n_input, 1, true),
    bias(1, n_output, 1, 1, false, random),
    dbias(1, n_output, 1, 1, true),
    vb(1, n_output, 1, 1, true)
{}

void Layer::Dense::forward()  {
    Mat const& input = get_input();
    set_output(mat_mul_vec(weight, input) + bias);
}

void Layer::Dense::backprop(Mat const& d_output) {
    Mat const& input = get_input();
    accumulate_v_mul_vT(dweight, d_output, input);
    dbias += d_output;
    accumulate_dinput(
        matT_mul_vec(weight, d_output)
    );
}

void Layer::Dense::flush_gradient() {
    static const Scalar beta = 0.9f;
    vw = lr * beta * vw + lr * (1.0f - beta) * dweight;
    vb = lr * beta * vb + lr * (1.0f - beta) * dbias;
    weight -= vw;
    bias   -= vb;
    dweight.fill_zero();
    dbias.fill_zero();
}


void Layer::Dense::dump_weight(std::string const& folder, std::string const& pre) {
    weight.dump(folder + "/" + pre + "w.bin");
    bias.dump(folder + "/" + pre + "b.bin");
}
void Layer::Dense::load_weight(std::string const& folder, std::string const& pre) {
    weight.load(folder + "/" + pre + "w.bin");
    bias.load(folder + "/" + pre + "b.bin");
}