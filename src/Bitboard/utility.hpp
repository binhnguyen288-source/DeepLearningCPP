#pragma once
#include "constants.hpp"

template<int shift>
BB check_dir(BB x) {
    
    constexpr BB filled_mask = []() {
        constexpr BB mask = []() {
            if constexpr (shift == bsize + 1)
                return not_first;
            else if constexpr (shift == bsize - 1) {
                return not_last;
            }
            else if constexpr (shift == 1) {
                return not_first;
            }
            else return draw_state;
        }();
        return mask & (mask << shift) & (mask << 2 * shift) & (mask << 3 * shift);
    }();
    BB const m = (x << 4 * shift) & filled_mask;
    x &= x << shift;
    x &= x << 2 * shift;
    return x & m;
}

bool check_win(BB x) {
    return (check_dir<bsize + 1>(x) | check_dir<bsize - 1>(x) | check_dir<bsize>(x) | check_dir<1>(x)) != 0;
}
static constexpr uint64_t hash_size = 25165843;
static constexpr uint64_t scale_upper = []() {
    uint64_t result = 1;
    for (int i = 0; i < bsize * bsize; ++i) {
        result = (result * 2) % hash_size;
    }
    return result;
}();

#include <optional>

struct hash_table {
    enum prob_type : uint8_t {
        NIL, EXACT, LOWER, UPPER
    };
    struct hash_entry {
        BB x{}, o{};
        float score{};
        prob_type type{NIL};
        uint8_t depth{};
        uint8_t move{};
    };
    hash_entry table[hash_size];
    std::optional<hash_entry> prob(BB x, BB o, uint8_t depth) {
        hash_entry ret = table[hash(x, o)];
        if (ret.type == NIL || ret.depth < depth || ret.x != x || ret.o != o) return std::nullopt;
        return ret;
    }
    void set_entry(BB x, BB o, float score, prob_type type, uint8_t depth, uint8_t move) {
        hash_entry& entry = table[hash(x, o)];
        if (depth >= entry.depth) {
            entry = {
                .x = x,
                .o = o,
                .score = score,
                .type = type,
                .depth = depth,
                .move = move,
            };
        }
    }
    void clear() {
        std::memset(table, 0, sizeof(table));
    }
    static uint64_t hash(BB x, BB o) {
        x %= hash_size;
        o %= hash_size;
        return (o * scale_upper + x) % hash_size;
    }
};