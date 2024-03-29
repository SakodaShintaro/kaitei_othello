﻿#pragma once

#ifndef PIECE_HPP
#define PIECE_HPP

#include<unordered_map>
#include<string>
#include<vector>
#include<iostream>
#include<fstream>
#include<cassert>
#include"types.hpp"
#include"array_map.hpp"

enum Piece : int32_t {
    BLACK_PIECE,
    WHITE_PIECE,
    EMPTY,
    WALL,
    PieceNum
};

extern const ArrayMap<std::string, PieceNum> PieceToStr;
extern const ArrayMap<std::string, PieceNum> PieceToSfenStr;

inline Piece operator++(Piece &p, int) { return p = static_cast<Piece>(p + 1); }
inline Piece operator|(Piece lhs, Piece rhs) { return Piece(int(lhs) | int(rhs)); }
inline int operator<<(Piece p, int shift) { return static_cast<int>(p) << shift; }

inline static Color operator~(Color c) { return (c == BLACK) ? WHITE : BLACK; }

inline Piece oppositeColor(Piece p) {
    return (p == BLACK_PIECE ? WHITE_PIECE : BLACK_PIECE);
}

std::ostream& operator<<(std::ostream&, const Piece piece);

#endif