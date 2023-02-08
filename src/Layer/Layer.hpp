#pragma once
#include "../Mat/Mat.hpp"
#include <cassert>
static constexpr Scalar lr = .001;

enum activation {
    TANH,
    SIGMOID,
    RELU,
    SOFTMAX
};

namespace Layer {

    class Layer {
    public:
        Layer(Size input_size, Size output_size) : 
            input(1, input_size, false), 
            output(1, output_size, false), 
            d_input(1, input_size, true),
            input_size{input_size},
            output_size{output_size} {}
        virtual ~Layer() = default;
        virtual void forward() = 0;
        virtual void backprop(Mat const& d_output) = 0;
        virtual void flush_gradient() = 0;
        virtual void dump_weight(std::string const& folder, std::string const& pre) = 0;
        virtual void load_weight(std::string const& folder, std::string const& pre) = 0;

        Size get_input_size()                   const   { return input_size; }
        Size get_output_size()                  const   { return output_size; }
        void set_input(Mat const& input) {
            assert(input.size == input_size); 
            this->input = input; 
        }
        void set_input(Mat&& input) { 
            assert(input.size == input_size);
            this->input = std::move(input); 
        }
        void accumulate_dinput(Mat const& add) { 
            assert(add.size == input_size);
            d_input += add; 
        }
        Mat const& get_output()                 const   { return output; }
        Mat const& get_input()                  const   { return input; }
        Mat get_dinput() {
            Mat copy(d_input);
            d_input.fill_zero();
            return copy;
        }
    protected:
        //void set_output(Mat const& output)              { this->output = output; }
        void set_output(Mat&& output) {
            assert(output.size == output_size); 
            this->output = std::move(output);
        }
    private:
        Mat input;
        Mat output;
        Mat d_input;
        Size input_size, output_size;
    };  
      
}