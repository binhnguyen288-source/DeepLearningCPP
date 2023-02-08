#include "Upscaling.hpp"

void Layer::UpscalingConcat::forward() {
    Mat const& input = get_input();
    Mat const& route_input = route_layer->get_input();
    Size const output_size = get_output_size();
    Size const input_size = get_input_size();
    Size const route_input_size = route_layer->get_input_size();
    //assert(route_input_size.channels == 2 * input_size.channels);

    Mat output(1, output_size, false);
    for (int i = 0; i < output_size.rows; ++i) {
        for (int j = 0; j < output_size.cols; ++j) {
            for (int c = 0; c < input_size.channels; ++c) {
                output(i, j, c) = input(i / 2, j / 2, c);
            }
            for (int c = 0; c < route_input_size.channels; ++c) {
                output(i, j, c + input_size.channels) = route_input(i, j, c);
            }
        }
    }

    set_output(std::move(output));
}
void Layer::UpscalingConcat::backprop(Mat const& d_output) {

    Size const output_size = get_output_size();
    Size const input_size = get_input_size();
    Size const route_input_size = route_layer->get_input_size();

    Mat d_input(1, input_size, true);
    Mat d_route_input(1, route_input_size, true);

    for (int i = 0; i < output_size.rows; ++i) {
        const int i_input = i / 2;
        for (int j = 0; j < output_size.cols; ++j) {
            const int j_input = j / 2;
            for (int c = 0; c < input_size.channels; ++c) {
                d_input(i_input, j_input, c) += d_output(i, j, c);
            }
            for (int c = 0; c < route_input_size.channels; ++c) {
                d_route_input(i, j, c) += d_output(i, j, c + input_size.channels);
            }
        }
    }
    
    route_layer->accumulate_dinput(d_route_input);
    accumulate_dinput(d_input);
}