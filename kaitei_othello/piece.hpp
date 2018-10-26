#pragma once

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

enum Piece{
    BLACK,
    WHITE,
    EMPTY,
    WALL,
    PieceNum
};

extern const ArrayMap<std::string, PieceNum> PieceToStr;
extern std::unordered_map<int32_t, std::string> PieceToSfenStr;

inline Piece operator++(Piece &p, int) { return p = static_cast<Piece>(p + 1); }
inline Piece operator|(Piece lhs, Piece rhs) { return Piece(int(lhs) | int(rhs)); }
inline int operator<<(Piece p, int shift) { return static_cast<int>(p) << shift; }

inline static Color operator~(Color c) { return (c == BLACK) ? WHITE : BLACK; }

extern const std::array<Piece, 28> PieceList;
extern const std::array<std::array<Piece, 14>, 2> ColoredPieceList;
extern const std::array<std::array<Piece, 3>, 2> ColoredJumpPieceList;
extern const ArrayMap<int32_t, PieceNum> pieceToIndex;

inline Piece oppositeColor(Piece p) {
    return (p == BLACK ? WHITE : BLACK);
}

std::ostream& operator<<(std::ostream&, const Piece piece);

#endif