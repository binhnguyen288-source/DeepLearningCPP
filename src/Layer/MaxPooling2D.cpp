#include "MaxPooling2D.hpp"


void Layer::MaxPooling2D::forward() {
    Size const input_size = get_input_size();
    Mat const& input = get_input();
    Mat output(1, get_output_size(), true);
    for (int i = 0; i < input_size.rows; ++i) {
        for (int j = 0; j < input_size.cols; ++j) {
            for (int c = 0; c < input_size.channels; ++c) {
                output(i / 2, j / 2, c) = std::max(output(i / 2, j / 2, c), input(i, j, c));
            }
        }
    }
    set_output(std::move(output));
}
void Layer::MaxPooling2D::backprop(Mat const& d_output) {
    Size const input_size = get_input_size();
    Mat const& input = get_input();
    Mat const& output = get_output();
    Mat d_input(1, get_input_size(), true);
    for (int i = 0; i < input_size.rows; ++i) {
        for (int j = 0; j < input_size.cols; ++j) {
            for (int c = 0; c < input_size.channels; ++c) {
                d_input(i, j, c) += input(i, j, c) == output(i / 2, j / 2, c) ? d_output(i / 2, j / 2, c) : 0.0f;
            }
        }
    }
    accumulate_dinput(d_input);
}