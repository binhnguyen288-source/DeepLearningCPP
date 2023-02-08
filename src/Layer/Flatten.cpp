#include "Flatten.hpp"

void Layer::Flatten::forward() {
    Mat const& input = get_input();
    Mat output(input);
    output.size = get_output_size();

    set_output(std::move(output));
}

void Layer::Flatten::backprop(Mat const& d_output) {
    Mat temp{d_output};
    temp.size = get_input_size();
    accumulate_dinput(temp);
}