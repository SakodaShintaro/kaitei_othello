#pragma once

#ifndef TYPES_HPP
#define TYPES_HPP

#include<iostream>
#include<array>

enum Color {
    BLACK, WHITE, ColorNum,
};

enum Bound {
    EXACT_BOUND, UPPER_BOUND, LOWER_BOUND
};

enum Depth {
    PLY = 128,
    DEPTH_MAX = 64 * PLY,
    MATE_DEPTH_MAX = 128,
};

using Score = float;
constexpr Score MAX_SCORE = 1000000.0;
constexpr Score SCORE_ZERO = 0;
constexpr Score DRAW_SCORE = 0;
constexpr Score MIN_SCORE = -MAX_SCORE;
constexpr Score MATE_SCORE_LOWER_BOUND = MAX_SCORE - static_cast<int>(MATE_DEPTH_MAX);
constexpr Score MATE_SCORE_UPPER_BOUND = MIN_SCORE + static_cast<int>(MATE_DEPTH_MAX);
constexpr Score SCORE_NONE = MAX_SCORE + 1;

constexpr double CP_GAIN = 1.0 / 600.0;

inline bool isMatedScore(const Score score) {
    return score <= MATE_SCORE_UPPER_BOUND || MATE_SCORE_LOWER_BOUND <= score;
}

//Depth
constexpr Depth operator-(Depth lhs) { return Depth(-int(lhs)); }
constexpr Depth operator+(Depth lhs, Depth rhs) { return Depth(int(lhs) + int(rhs)); }
constexpr Depth operator-(Depth lhs, Depth rhs) { return Depth(int(lhs) - int(rhs)); }
constexpr Depth operator+(Depth lhs, int rhs) { return Depth(int(lhs) + rhs); }
constexpr Depth operator-(Depth lhs, int rhs) { return Depth(int(lhs) - rhs); }
constexpr Depth operator+(int lhs, Depth rhs) { return Depth(lhs + int(rhs)); }
constexpr Depth operator-(int lhs, Depth rhs) { return Depth(lhs - int(rhs)); }
inline Depth& operator+=(Depth& lhs, Depth rhs) { return lhs = lhs + rhs; }
inline Depth& operator-=(Depth& lhs, Depth rhs) { lhs = lhs - rhs;  return lhs; }
inline Depth& operator+=(Depth& lhs, int rhs) { lhs = lhs + rhs;  return lhs; }
inline Depth& operator-=(Depth& lhs, int rhs) { lhs = lhs - rhs;  return lhs; }
inline Depth& operator++(Depth& lhs) { lhs = lhs + 1;  return lhs; }
inline Depth& operator--(Depth& lhs) { lhs = lhs - 1;  return lhs; }
inline Depth operator++(Depth& lhs, int) { Depth t = lhs; lhs += 1;  return t; }
inline Depth operator--(Depth& lhs, int) { Depth t = lhs; lhs -= 1;  return t; }

constexpr Depth operator*(Depth lhs, int rhs) { return Depth(int(lhs) * rhs); }
constexpr Depth operator*(int lhs, Depth rhs) { return Depth(lhs * int(rhs)); }
constexpr Depth operator/(Depth lhs, int rhs) { return Depth(int(lhs) / rhs); }
inline Depth& operator*=(Depth& lhs, int rhs) { lhs = lhs * rhs;  return lhs; }
inline Depth& operator/=(Depth& lhs, int rhs) { lhs = lhs / rhs;  return lhs; }
std::ostream& operator<<(std::ostream& os, const Depth d);
std::istream& operator>>(std::istream& is, Depth& d);

//std::arrayに関するオーバーロード
inline std::array<int32_t, 2>& operator+=(std::array<int32_t, 2>& lhs, std::array<int16_t, 2> rhs) {
    lhs[0] += rhs[0];
    lhs[1] += rhs[1];
    return lhs;
}
inline std::array<int32_t, 2>& operator-=(std::array<int32_t, 2>& lhs, std::array<int16_t, 2> rhs) {
    lhs[0] -= rhs[0];
    lhs[1] -= rhs[1];
    return lhs;
}
template<class T>
inline std::array<T, 2>& operator+=(std::array<T, 2>& lhs, std::array<T, 2> rhs) {
    lhs[0] += rhs[0];
    lhs[1] += rhs[1];
    return lhs;
}
template<class T>
inline std::array<T, 2>& operator-=(std::array<T, 2>& lhs, std::array<T, 2> rhs) {
    lhs[0] -= rhs[0];
    lhs[1] -= rhs[1];
    return lhs;
}
inline std::array<int16_t, 2> operator*(int c, std::array<int16_t, 2> rhs) {
    rhs[0] *= c;
    rhs[1] *= c;
    return rhs;
}
inline std::array<int32_t, 2> operator*(int c, std::array<int32_t, 2> rhs) {
    rhs[0] *= c;
    rhs[1] *= c;
    return rhs;
}

//これで上の要らなくなりそうだけど
template<class T, size_t SIZE>
inline std::array<T, SIZE> operator+(std::array<T, SIZE> lhs, std::array<T, SIZE> rhs) {
    for (size_t i = 0; i < SIZE; i++) {
        lhs[i] += rhs[i];
    }
    return lhs;
}

template<class T, size_t SIZE>
inline std::array<T, SIZE>& operator+=(std::array<T, SIZE>& lhs, std::array<T, SIZE> rhs) {
    for (size_t i = 0; i < SIZE; i++) {
        lhs[i] += rhs[i];
    }
    return lhs;
}

template<class T, class U, size_t SIZE>
inline std::array<T, SIZE> operator/(std::array<T, SIZE> lhs, U rhs) {
    for (size_t i = 0; i < SIZE; i++) {
        lhs[i] /= rhs;
    }
    return lhs;
}

template<class T, class U, size_t SIZE>
inline std::array<T, SIZE>& operator/=(std::array<T, SIZE>& lhs, U rhs) {
    for (size_t i = 0; i < SIZE; i++) {
        lhs[i] /= rhs;
    }
    return lhs;
}

#endif // !TYPES_HPP