#pragma once
#ifndef HISTORY_HPP
#define HISTORY_HPP

#include"piece.hpp"
#include"square.hpp"
#include"move.hpp"
#include<queue>

class History {
private:
    Score table_[SquareNum]; //4 * 88 * 121 * 2 = 85KB
public:
    void updateBetaCutMove(const Move move, const Depth depth) {
        int32_t d = depth / PLY;
        (*this)[move] += (d > 17 ? 0 : d * d + 2 * d - 2);
    }

    void updateNonBetaCutMove(const Move move, const Depth depth) {
        int d = depth / PLY;
        (*this)[move] -= (d > 17 ? 0 : d * d + 2 * d - 2);
    }

    Score operator[](const Move& move) const {
        return table_[move.to()];
    }
    Score& operator[](const Move& move) {
        return table_[move.to()];
    }

    void clear() {
        for (int to = 0; to < SquareNum; ++to) {
            table_[to] = SCORE_ZERO;
            table_[to] = SCORE_ZERO;
        }
    }
};

#endif // !HISTORY_HPP