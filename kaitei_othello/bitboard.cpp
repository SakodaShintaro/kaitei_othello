#include"bitboard.hpp"

Bitboard BOARD_BB;
Bitboard SQUARE_BB[SquareNum];
Bitboard FILE_BB[FileNum];
Bitboard RANK_BB[RankNum];
Bitboard BETWEEN_BB[SquareNum][SquareNum];

std::ostream& operator << (std::ostream& os, const Bitboard& rhs) {
    for (int r = Rank1; r <= Rank9; ++r) {
        for (int f = File9; f >= File1; --f) {
            Bitboard target = rhs & SQUARE_BB[FRToSquare[f][r]];
            os << (target ? " *" : " .");
        }
        os << std::endl;
    }
    return os;
}

void Bitboard::init() {
    //1.SQUARE_BB
    for (auto sq : SquareList) {
        SQUARE_BB[sq] = Bitboard(sq);
        BOARD_BB |= SQUARE_BB[sq];
    }

    //2.FILE_BB,RANK_BB
    for (int f = File1; f <= File9; ++f) {
        for (int r = Rank1; r <= Rank9; ++r) {
            FILE_BB[f] |= SQUARE_BB[FRToSquare[f][r]];
            RANK_BB[r] |= SQUARE_BB[FRToSquare[f][r]];
        }
    }

    //BETWEEN_BB
    for (Square sq1 : SquareList) {
        for (Square sq2 : SquareList) {
            auto dir = directionAtoB(sq1, sq2);
            if (dir == H)
                continue;
            //1マスずつたどっていく
            for (Square between = sq1 + dir; between != sq2; between = between + dir) {
                BETWEEN_BB[sq1][sq2] |= SQUARE_BB[between];
            }
        }
    }
}