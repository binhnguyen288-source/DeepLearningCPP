#pragma once
#include "Layer/Layers.hpp"
#include "Layer/Sequential.hpp"
#include "iou.hpp"
constexpr int yolo_size = 320;
constexpr int num_classes = 20;
constexpr std::array<int, 3> scale_size{
    yolo_size / 32,
    yolo_size / 16,
    yolo_size / 8
};
constexpr std::array<Rect, 9> anchors{
    Rect{0.28f, 2.22f}, Rect{0.38f, 0.48f}, Rect{0.9f, 0.78f},
    Rect{0.07f, 0.15f}, Rect{0.15f, 0.11f}, Rect{0.14f, 0.29f},
    Rect{0.02f, 0.03f}, Rect{0.04f, 0.07f}, Rect{0.08f, 0.06f}
};
struct YoloV3 {
    Sequential net;
    // Sequential branch1;
    // Sequential branch2;
    YoloV3() : net(Size(yolo_size, yolo_size, 3), Sequential::SUM_SQUARES_ERROR) {
        net.add_conv2d_layer(32, 3, 1, 1);
        net.add_conv2d_layer(64, 3, 2, 1);
        net.add_residual_layer();
        net.add_conv2d_layer(128, 3, 2, 1);
        net.add_residual_layer();
        net.add_residual_layer();
        net.add_conv2d_layer(256, 3, 2, 1);
        net.add_residual_layer();
        net.add_residual_layer();
        net.add_residual_layer();
        net.add_residual_layer();
        net.add_residual_layer();
        net.add_residual_layer();
        net.add_residual_layer();
        net.add_residual_layer();
        auto route2 = net.add_conv2d_layer(512, 3, 2, 1);
        net.add_residual_layer();
        net.add_residual_layer();
        net.add_residual_layer();
        net.add_residual_layer();
        net.add_residual_layer();
        net.add_residual_layer();
        net.add_residual_layer();
        net.add_residual_layer();
        auto route1 = net.add_conv2d_layer(1024, 3, 2, 1);
        net.add_residual_layer();
        net.add_residual_layer();
        net.add_residual_layer();
        net.add_residual_layer();
        net.add_conv2d_layer(512, 1, 1, 0);
        net.add_conv2d_layer(1024, 3, 1, 1); // ... branch scale prediction 1
        // branch1 = Sequential(net.get_layer_view(), net.get_input_size());

        net.add_conv2d_layer(256, 1, 1, 0);
        net.add_concat_layer(route1);       // .. upscaling concat 1
        net.add_conv2d_layer(256, 1, 1, 0);
        net.add_conv2d_layer(512, 3, 1, 1); // ... branch scale prediction 2

        // branch2 = Sequential(net.get_layer_view(), net.get_input_size());
        
        net.add_conv2d_layer(128, 1, 1, 0);
        net.add_concat_layer(route2);
        net.add_conv2d_layer(128, 1, 1, 0);
        net.add_conv2d_layer(256, 3, 1, 1); // ... branch scale prediction 3
        net.add_conv2d_layer(512, 3, 1, 1);
        net.add_conv2d_layer((num_classes + 5) * 3, 1, 1, 0);

        // branch1.add_conv2d_layer(2048, 3, 1, 1);
        // branch1.add_conv2d_layer((num_classes + 5) * 3, 1, 1, 0);

        // branch2.add_conv2d_layer(1024, 3, 1, 1);
        // branch2.add_conv2d_layer((num_classes + 5) * 3, 1, 1, 0);
    }
};
