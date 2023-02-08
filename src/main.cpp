


#include <iostream>
#include <cstring>
#include <chrono>

#include "Layer/Sequential.hpp"
#include "Bitboard/transform.hpp"
#include "Bitboard/constants.hpp"
#include "Bitboard/utility.hpp"





Sequential model = []() {
    Sequential model(Size(bsize, bsize, 1), Sequential::SUM_SQUARES_ERROR);
    model.add_conv2d_layer(16, 4, 1, 0);
    model.add_conv2d_layer(8, 3, 1, 0);
    model.add_maxpooling2d_layer();
    model.add_flatten_layer();
    model.add_dense_layer(128, RELU);
    model.add_dense_layer(1, TANH);
    return model;
}();

Mat get_input_dnn(BB x, BB o) {
    Mat input(bsize, bsize, 1, false);
    for (int i = 0; i < bsize * bsize; ++i) {
        if (x >> i & 1) input[i] = 1.0f;
        else if (o >> i & 1) input[i] = -1.0f;
        else input[i] = 0.0f;
    }
    return input;
}

inline uint8_t get_move_pos(BB x) {
    uint64_t flat[sizeof(x) / 8];
    std::memcpy(flat, &x, sizeof(x));
    return flat[0] ? __builtin_ctzll(flat[0]) : (64 + __builtin_ctzll(flat[1]));
}

Scalar evaluate(BB x, BB o) {
    //return 0;
    return model.infer(get_input_dnn(x, o))[0];
}

Scalar direct_evaluate(Mat const& state) {
    return model.infer(state)[0];
}

BB KH[bsize * bsize];
hash_table entries;


float search(BB x, BB o, float alpha, float beta, int depth, int ply, Mat& us, Mat& them) {
    if (check_win(o)) return -1.0f;
    if ((x | o) == draw_state) return 0.0f;
    const float alpha_original = alpha;
    auto prob = entries.prob(x, o, depth);
    std::array<BB, 2> good_move_test{};
    int good_move_count = 0;
    BB empty = draw_state & ~(x | o);
    if (prob.has_value()) {
        auto entry = *prob;
        switch (entry.type) {

            case hash_table::EXACT:
                return entry.score;
                break;
            case hash_table::LOWER:
                alpha = std::max(alpha, entry.score);
                break;
            case hash_table::UPPER:
                beta = std::min(beta, entry.score);
                break;
            default:
                break;
        }
        if (alpha >= beta)
            return entry.score;

        if (entry.type == hash_table::EXACT || entry.type == hash_table::LOWER) {
            BB move = (BB)1 << entry.move;
            if (move & empty) good_move_test[good_move_count++] = move;
        }
    }
    if (depth == 0) return direct_evaluate(us);
    
    float best = -1.0f;
    BB best_move = empty & -empty;
    if (KH[ply] & empty) good_move_test[good_move_count++] = KH[ply];
    for (int i = 0; i < good_move_count; ++i) {
        BB const move = good_move_test[i];
        uint8_t idx = get_move_pos(move);
        us[idx] = 1.0f;
        them[idx] = -1.0f;
        const float score = -search(o, x ^ move, -beta, -alpha, depth - 1, ply + 1, them, us);
        us[idx] = 0.0f;
        them[idx] = 0.0f;
        if (score > best) {
            best = score;
            best_move = move;
            if (score > alpha) {
                alpha = score;
                if (score >= beta) {
                    KH[ply] = move;
                    goto save_entry;
                }
            }
        }
        empty ^= move;
    }
    while (empty) {
        BB const move = empty & -empty;
        uint8_t idx = get_move_pos(move);
        us[idx] = 1.0f;
        them[idx] = -1.0f;
        const float score = -search(o, x ^ move, -beta, -alpha, depth - 1, ply + 1, them, us);
        us[idx] = 0.0f;
        them[idx] = 0.0f;
        if (score > best) {
            best = score;
            best_move = move;
            if (score > alpha) {
                alpha = score;
                if (score >= beta) {
                    KH[ply] = move;
                    goto save_entry;
                }
            }
        }
        empty ^= move;
    }
    
save_entry:

    if (best <= alpha_original) {
        entries.set_entry(x, o, best, hash_table::UPPER, depth, get_move_pos(best_move));
    }
    else if (best >= beta) {
        entries.set_entry(x, o, best, hash_table::LOWER, depth, get_move_pos(best_move));
    }
    else {
        entries.set_entry(x, o, best, hash_table::EXACT, depth, get_move_pos(best_move));
    }
   
    return best;
}


#define web 0
#if web == 1

BB x, o;
int get_move_pos(BB x) {
    for (int i = 0; i < bsize * bsize; ++i) {
        if (x >> i & 1) return i;
    }
    return bsize * bsize;
}

extern "C" void init_weights() {
    model.load_weight("weights");

}
extern "C" void clear() {
    x = o = 0;
}
enum {
    machine_win = -2,
    human_win = -1
};
extern "C" int get_next_move(int move) {
    using namespace std::chrono;
    std::memset(KH, 0, sizeof(KH));
    
    if (check_win(o)) {
        return machine_win;
    }
    x |= (BB)1 << move;
    if (check_win(x)) {
        return human_win;
    }
    auto s = steady_clock::now();
    int depth = 1;
    float score = 0;
    for (;; ++depth) {
        score = search(o, x, -1.0f, 1.0f, depth, 0);
        if (score >= 1.0f || 
            score <= -1.0f || 
            (depth >= 4 && (steady_clock::now() - s) > milliseconds{500}))
            break;
    }
    o |= best_move;
    if (check_win(o)) {
        return machine_win;
    }

    return get_move_pos(best_move);
}
#else

void print(BB x, BB o) {
    for (int i = 0; i < bsize; ++i) {
        for (int j = 0; j < bsize; ++j) {
            int idx = bsize * i + j;
            if (x >> idx & 1) {
                std::cout << 'x';
            }
            else if (o >> idx & 1) {
                std::cout << 'o';
            }
            else std::cout << '.';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}
int main() {
    constexpr float vanishing = 0.95f;
    model.load_weight("weights");


    std::vector<Mat> all_states;
    std::vector<Mat> all_scores;
    for (int n_matches = 0;; ++n_matches) {
        if (n_matches % 5 == 0) {
            std::cout << "Saving weights\n";
            model.dump_weight("weights");
        }
        std::vector<std::pair<BB, BB>> states;
        
        BB boards[2] = {0};
        auto& [x, o] = boards;
        float end_score{};
        for (int i = 0;; i ^= 1) {
            print(x, o);
            states.push_back({boards[i], boards[i ^ 1]});
            if (check_win(boards[i ^ 1])) {
                std::cout << (i ? 'X' : 'O') << " wins\n";
                end_score = -1.0f;
                break;
            }
            if ((x | o) == draw_state) {
                
                std::cout << "Draw\n";
                end_score = 0.0f;
                break;
            }
            std::memset(KH, 0, sizeof(KH));
            using namespace std::chrono;
            auto s = steady_clock::now();
            int depth = 1;
            float score{};
            entries.clear();
            Mat us = get_input_dnn(boards[i], boards[i ^ 1]);
            Mat them = get_input_dnn(boards[i ^ 1], boards[i]);
            for (;; ++depth) {
                score = search(boards[i], boards[i ^ 1], -1.0f, 1.0f, depth, 0, us, them);
                if (score >= 1.0f || 
                    score <= -1.0f || 
                    (depth >= 4 && (steady_clock::now() - s) > milliseconds{500}))
                    break;
            }
            BB best_move = (BB)1 << entries.prob(boards[i], boards[i ^ 1], 0)->move;
            boards[i] ^= best_move;
            std::cout << "Depth: " << depth << std::endl;
            std::cout << "Score: " << score << std::endl;
            std::cout << "Match: " << n_matches << std::endl;
        }
        std::vector<Mat> arg_states;
        std::vector<Mat> scores;
        for (int i = states.size() - 1; i >= 0; --i) {
            auto temp = permute(states[i].first, states[i].second);
            for (const auto& p : temp) {
                auto [x, o] = p;
                Mat s(1, 1, 1, false);
                s[0] = end_score;
                
                scores.push_back(std::move(s));
                arg_states.push_back(get_input_dnn(x, o));
            }
            if (i > 0)
                end_score = model.infer(get_input_dnn(states[i - 1].first, states[i - 1].second))[0] * (1.0f - vanishing) - vanishing * end_score;
        }
        model.backprop(arg_states, scores);
        
    }

    return 0;
}

#endif
