#pragma once
#include "Mat/Mat.hpp"
struct Rect {
    Scalar x, y, w, h;
    constexpr Rect(Scalar x, Scalar y, Scalar w, Scalar h) : x{x}, y{y}, w{w}, h{h} {}
    constexpr Rect(Scalar w, Scalar h) : x{}, y{}, w{w}, h{h} {}
    constexpr Scalar min_x() const {
        return x - w * 0.5f;
    }
    constexpr Scalar max_x() const{
        return x + w * 0.5f;
    }
    constexpr Scalar min_y() const {
        return y - h * 0.5f;
    }
    constexpr Scalar max_y() const {
        return y + h * 0.5f;
    }
    constexpr Scalar area() const {
        return w * h;
    }
};


constexpr Scalar IoU(Rect const& a, Rect const& b) {

    Scalar dx = std::max(std::min(a.max_x(), b.max_x()) - std::max(a.min_x(), b.min_x()), 0.0f);
    Scalar dy = std::max(std::min(a.max_y(), b.max_y()) - std::max(a.min_y(), b.min_y()), 0.0f);
    Scalar area_intersect = dx * dy;
    Scalar area_union = a.area() + b.area() - area_intersect;

    return area_intersect / area_union;

}

static_assert(IoU(Rect{1, 1, 1, 1}, Rect{1, 1, 2, 1}) == 0.5f);


