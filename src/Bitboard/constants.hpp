#pragma once
using BB = __uint128_t;
constexpr int bsize = 11;

constexpr BB draw_state = []() {
    BB result{};
    for (int i = 0; i < bsize * bsize; ++i) {
        result |= (BB)1 << i;
    }
    return result;
}();

constexpr BB not_first = []() {
    BB first{};
    for (int i = 0; i < bsize * bsize; i += bsize) {
        first |= (BB)1 << i;
    }
    return draw_state & ~first;
}();

constexpr BB not_last = []() {
    BB last{};
    for (int i = bsize - 1; i < bsize * bsize; i += bsize) {
        last |= (BB)1 << i;
    }
    return draw_state & ~last;
}();