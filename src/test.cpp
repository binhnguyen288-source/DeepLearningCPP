#include "Layer/Sequential.hpp"
#include "MNIST/mnist.hpp"
#include <any>
#include "yolov3.hpp"

void run_test() {

    Sequential model(Size(28, 28, 1), Sequential::loss_function::CATEGORICAL_CROSSENTROPY);
    
    auto route = model.add_conv2d_layer(32, 3, 1, 1, activation::RELU);
    model.add_maxpooling2d_layer();
    // model.add_residual_layer();
    // model.add_residual_layer();
    // model.add_residual_layer();
    // model.add_concat_layer(route);
    // model.add_residual_layer();
    model.add_conv2d_layer(32, 3, 1, 1);
    model.add_maxpooling2d_layer();
    // model.add_residual_layer();
    // model.add_conv2d_layer(32, 3, 1, 1);
    // model.add_maxpooling2d_layer();
    model.add_flatten_layer();
    model.add_dense_layer(128, activation::RELU);
    model.add_dense_layer(10,  activation::SOFTMAX);

    auto training_image{read_mnist_images("training_image.bin")};
    auto training_label{read_mnist_labels("training_label.bin")};


    auto testing_image{read_mnist_images("testing_image.bin")};
    auto testing_label{read_mnist_labels("testing_label.bin")};
    YoloV3 test;
    test.net.add_dense_layer(10, SOFTMAX);
    test.net.backprop_test(training_image, training_label, testing_image, testing_label);
}



#include <fstream>
#include <chrono>
#include "iou.hpp"



Scalar sigmoid(Scalar v) {
    double x = v;
    x = std::clamp(x, -15.0, 15.0);

    if (x >= 0.0) 
        return 1.0 / (1.0 + std::exp(x));

    double e = std::exp(x);
    
    return e / (e + 1.0);
}


Scalar mylog(Scalar v) {
    return std::log(v + 0.00001);
}


Mat yolo_gradient(Mat const& output, Mat const& target, int scale_idx, Scalar& loss) {
    const int scale = scale_size[scale_idx];
    // anchors[scale_idx][0];
    constexpr Scalar lnoobj = 0.1f;
    constexpr Scalar lobj = 1.0f;
    Mat d_input(1, scale, scale, 3 * (num_classes + 5), true);
    for (int i = 0; i < scale; ++i) {
        for (int j = 0; j < scale; ++j) {
            
            
            for (int box_idx = 0; box_idx < 3; ++box_idx) {

                int const box_offset = (5 + num_classes) * box_idx;

                static const auto square = [](Scalar x) { return x * x; };
                
                Scalar const p_object = sigmoid(output(i, j, box_offset));

                if (std::fabs(target(i, j, box_offset)) <= 1e-8) {

                    // no object loss
                    
                    d_input(i, j, box_offset) = lnoobj * (p_object - 0);
                    loss += -lnoobj * mylog(1.0f - p_object);
                }

                else {
                    Rect truth{
                        target(i, j, box_offset + 1),
                        target(i, j, box_offset + 2),
                        target(i, j, box_offset + 3),
                        target(i, j, box_offset + 4)
                    };

                    Rect box{
                        
                        sigmoid( output(i, j, box_offset + 1)),
                        sigmoid( output(i, j, box_offset + 2)),
                        std::exp(output(i, j, box_offset + 3)) * anchors[scale_idx * 3 + box_idx].w,
                        std::exp(output(i, j, box_offset + 4)) * anchors[scale_idx * 3 + box_idx].h
                        
                    };
                    Scalar iou = IoU(box, truth);
                    loss += lobj * (
                        square(box.x - truth.x) + 
                        square(box.y - truth.y) + 
                        square(box.w - truth.w) + 
                        square(box.h - truth.h) -
                        mylog(p_object)
                    );
                    d_input(i, j, box_offset)       = lobj * (p_object - 1.0f);
                    d_input(i, j, box_offset + 1)   = lobj * (box.x - box.x * box.x) * (box.x - truth.x);
                    d_input(i, j, box_offset + 2)   = lobj * (box.y - box.y * box.y) * (box.y - truth.y);
                    d_input(i, j, box_offset + 3)   = lobj * (output(i, j, box_offset + 3) - std::log(truth.w / anchors[scale_idx * 3 + box_idx].w));
                    d_input(i, j, box_offset + 4)   = lobj * (output(i, j, box_offset + 4) - std::log(truth.h / anchors[scale_idx * 3 + box_idx].h));

                    for (int cur_class = 0; cur_class < num_classes; ++cur_class) {

                        Scalar prob_class = sigmoid(output(i, j, box_offset + 5 + cur_class));
                        Scalar prob_target = target(i, j, box_offset + 5 + cur_class);
                        d_input(i, j, box_offset + 5 + cur_class) = lobj * (prob_class - prob_target);
                        loss += -lobj * (prob_target * mylog(prob_class) + (1.0f - prob_target) * mylog(1.0f - prob_class));
                    }


                }
            }
        }
    }
    //std::cout << "Loss: " << loss << std::endl;
    return d_input;
}


constexpr int max_image = 1000;
#define load_image 0
#if load_image == 1
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
void load_image_voc() {
    std::ifstream train("voc/train.csv");
    std::string line;

    Mat training_images(
        max_image,
        yolo_size,
        yolo_size,
        3,
        true
    );

    std::array<Mat, 3> training_labels{
        Mat(max_image, scale_size[0], scale_size[0], (5 + num_classes) * 3, true),
        Mat(max_image, scale_size[1], scale_size[1], (5 + num_classes) * 3, true),
        Mat(max_image, scale_size[2], scale_size[2], (5 + num_classes) * 3, true),
    };
    
    int count = 0;
    while (std::getline(train, line)) {
        int const split_point = line.find(',');
        
        std::string const image_path(line.substr(0, split_point));
        std::string const label_path(line.substr(split_point + 1));


        {
            cv::Mat image = cv::imread("voc/images/" + image_path);
            cv::resize(image, image, cv::Size(yolo_size, yolo_size));
            if (!image.isContinuous()) image = image.clone();
            std::copy(image.data, image.data + 3 * yolo_size * yolo_size, &training_images(count, 0, 0, 0));
        }


        // std::array<Mat, 3> targets{
        //     Mat(1, scale_size[0], scale_size[0], (5 + num_classes) * 3, true),
        //     Mat(1, scale_size[1], scale_size[1], (5 + num_classes) * 3, true),
        //     Mat(1, scale_size[2], scale_size[2], (5 + num_classes) * 3, true),
        // };

        std::ifstream label_file("voc/labels/" + label_path);
        int class_label = 0;
        float x, y, width, height;
        while (label_file >> class_label >> x >> y >> width >> height) {
            std::array<int, 9> anchor_indexs {
                0, 1, 2, 3, 4, 5, 6, 7, 8
            };

            std::sort(anchor_indexs.begin(), anchor_indexs.end(), [bbox = Rect{width, height}](int i, int j) {
                return IoU(bbox, anchors[i]) > IoU(bbox, anchors[j]);
            });

            std::array<bool, 3> has_anchor{false, false, false};

            for (int anchor_idx : anchor_indexs) {
                int scale_idx       = anchor_idx / 3;
                int anchor_on_scale = anchor_idx % 3;
                int scale           = scale_size[scale_idx];
                int i               = floorf(y * scale);
                int j               = floorf(x * scale);

                bool anchor_taken = training_labels[scale_idx](count, i, j, anchor_on_scale * (5 + num_classes)) > 0.0f;

                if (!anchor_taken && !has_anchor[scale_idx]) {
                    training_labels[scale_idx](count, i, j, anchor_on_scale * (5 + num_classes)) = 1.0f;
                    training_labels[scale_idx](count, i, j, anchor_on_scale * (5 + num_classes) + 1) = scale * x - j;
                    training_labels[scale_idx](count, i, j, anchor_on_scale * (5 + num_classes) + 2) = scale * y - i;
                    training_labels[scale_idx](count, i, j, anchor_on_scale * (5 + num_classes) + 3) = scale * width;
                    training_labels[scale_idx](count, i, j, anchor_on_scale * (5 + num_classes) + 4) = scale * height;
                    training_labels[scale_idx](count, i, j, anchor_on_scale * (5 + num_classes) + 5 + class_label) = 1.0f;
                    has_anchor[scale_idx] = true;
                }
            }
        }

        if (++count == max_image) break;
    }

    training_images.dump("training.bin");
    
    training_labels[0].dump("testing0.bin");
    training_labels[1].dump("testing1.bin");
    training_labels[2].dump("testing2.bin");
}
#endif

int main() {
    #if load_image == 1
    load_image_voc();
    #endif

    using namespace std::chrono;

    
    Mat training_images(
        max_image,
        yolo_size,
        yolo_size,
        3,
        true
    );

    std::array<Mat, 3> training_labels{
        Mat(max_image, scale_size[0], scale_size[0], (5 + num_classes) * 3, true),
        Mat(max_image, scale_size[1], scale_size[1], (5 + num_classes) * 3, true),
        Mat(max_image, scale_size[2], scale_size[2], (5 + num_classes) * 3, true),
    };

    training_images.load("training.bin");
    training_labels[2].load("testing2.bin");

    

    YoloV3 yolo;
    yolo.net.load_weight("yolo_weight");
    for (int epoch = 0;; ++epoch) {
        std::cout << "Starting epoch " << epoch << std::endl;
        for (int scale_idx = 2; scale_idx < 3; ++scale_idx) {
            
            for (int i = 0; i < training_images.batches; ++i) {
                Scalar loss{};
                std::cout << "Infering..." << i << std::endl;
                Mat image(1, yolo_size, yolo_size, 3, false);
                std::copy(&training_images(i, 0, 0, 0), &training_images(i, 0, 0, 0) + image.total(), image.begin());
                Mat output = yolo.net.infer(image);
                std::cout << "Backproping..." << std::endl;
                int S = scale_size[scale_idx];
                Mat label(1, S, S, (5 + num_classes) * 3, false);
                auto start = &training_labels[scale_idx](i, 0, 0, 0);
                std::copy(start, start + label.total(), label.begin());
                Mat d_output = yolo_gradient(output, label, scale_idx, loss);
                yolo.net.backprop_doutput(d_output);
                std::cout << "Loss: " << loss << std::endl;
                
                yolo.net.flush_gradients();
            }
        }
        std::cout << "Saving weight\n";
        yolo.net.dump_weight("yolo_weight");
    }



    
    return 0;
}