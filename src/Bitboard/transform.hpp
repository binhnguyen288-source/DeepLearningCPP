#pragma once

#include "constants.hpp"

BB flip_vertical(BB x) {
    BB result{};
    for (int i = 0; i < bsize * bsize; ++i) {
        int row = i / bsize;
        int col = i % bsize;
        int idx = row * bsize + (bsize - 1 - col);
        result |= ((x >> i) & 1) << idx;
    }
    return result;
}

BB flip_horizontal(BB x) {
    BB result{};
    for (int i = 0; i < bsize * bsize; ++i) {
        int row = i / bsize;
        int col = i % bsize;
        int idx = (bsize - 1 - row) * bsize + col;
        result |= ((x >> i) & 1) << idx;
    }
    return result;
}

BB transpose(BB x) {
    BB result{};
    for (int i = 0; i < bsize * bsize; ++i) {
        int row = i / bsize;
        int col = i % bsize;
        int idx = col * bsize + row;
        result |= ((x >> i) & 1) << idx;
    }
    return result;
}

std::vector<std::pair<BB, BB>> permute(BB x, BB o) {
    static constexpr std::array trans{
        flip_horizontal,
        flip_vertical,
        transpose
    };
    std::vector<std::pair<BB, BB>> result;
    result.push_back({x, o});
    for (auto const& tran : trans) {
        
        result.push_back({ tran(x), tran(o) });
    }
    for (int i = 0; i < trans.size(); ++i) {
        for (int j = 0; j < trans.size(); ++j) {
            if (i == j) continue;
            result.push_back({trans[i](trans[j](x)), trans[i](trans[j](o))});
        }
    }
    std::sort(result.begin(), result.end());

    auto end_it = std::unique(result.begin(), result.end());
    result.erase(end_it, result.end());
    return result;
}