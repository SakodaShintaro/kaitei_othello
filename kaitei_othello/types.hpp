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
    DEPTH_MAX = 128 * PLY
};

using Score = float;
constexpr Score MAX_SCORE = 1.0f;
constexpr Score SCORE_ZERO = 0.0f;
constexpr Score DRAW_SCORE = 0.0f;
constexpr Score MIN_SCORE = 0.0f;

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