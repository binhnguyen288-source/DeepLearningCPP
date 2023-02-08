#include "ResidualBlock.hpp"


void Layer::ResidualBlock::forward() {
    Mat const& input = get_input();
    w1.set_input(input);
    w1.forward();
    Mat inter_input = w1.get_output().apply([](Scalar x) {
        return x > 0.0f ? x : 0.1f * x;    
    });
    w2.set_input(std::move(inter_input));
    w2.forward();
    set_output(input + w2.get_output());
}

void Layer::ResidualBlock::backprop(Mat const& d_output) {
    w2.backprop(d_output);
    Mat drelu = w1.get_output().apply([](Scalar x) {
        return x > 0.0f ? 1.0f : 0.1f;
    }) * w2.get_dinput();
    w1.backprop(drelu);
    accumulate_dinput(d_output + w1.get_dinput());
}

void Layer::ResidualBlock::flush_gradient() {
    w1.flush_gradient();
    w2.flush_gradient();
}

void Layer::ResidualBlock::dump_weight(std::string const& folder, std::string const& pre) {
    w1.dump_weight(folder, pre + "1");
    w2.dump_weight(folder, pre + "2");
}
void Layer::ResidualBlock::load_weight(std::string const& folder, std::string const& pre) {
    
    w1.load_weight(folder, pre + "1");
    w2.load_weight(folder, pre + "2");
}