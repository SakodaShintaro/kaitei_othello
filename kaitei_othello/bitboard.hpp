#pragma once
#ifndef BITBOARD_HPP
#define BITBOARD_HPP

#include"square.hpp"
#include"common.hpp"
#include<bitset>

class Bitboard {
public:
	//引数なしコンストラクタは空でいいかな
	Bitboard() {}

	//値を直接引数に持つコンストラクタ
    Bitboard(uint64_t b) : board_(b) {}

	//Squareを指定してそこだけを立てるコンストラクタ
	Bitboard(Square sq) {
		board_ = (1LL << SquareToNum[sq]);
	}

	//Stockfishとの互換性がなんちゃら
	//普通にあった方が便利そうだけども
    operator bool() const {
        return !(board_ == 0);
    }

    Square pop() {
        Square sq = SquareList[pop_lsb(board_)];
        assert(isOnBoard(sq));
        return sq;
    }

    auto pop_count() const {
        return POP_CNT64(board_);
    }

    template<class Function>
    void forEach(Function f) const {
        Bitboard copy = *this;
        while (copy) {
            Square sq = copy.pop();
            f(sq);
        }
    }

	//演算子類
	Bitboard operator ~() const {
        return Bitboard(~board_);
    }
	Bitboard operator |(const Bitboard& bb) const {
		return Bitboard(board_ | bb.board_);
	}
	Bitboard operator |(const Square sq) {
		return *this | Bitboard(sq);
	}
    Bitboard operator &(const Bitboard& bb) const {
        return Bitboard(board_ & bb.board_);
    }
    Bitboard& operator|=(const Bitboard& rhs) {
        board_ |= rhs.board_;
        return *this;
    }
    Bitboard& operator&=(const Bitboard& rhs) {
        board_ &= rhs.board_;
        return *this;
    }
    Bitboard& operator <<= (const int shift) {
        board_ <<= shift;
        return *this;
    }
    Bitboard operator << (const int shift) {
        return Bitboard(*this) <<= shift;
    }

    static void init();

    uint64_t board_;
};

extern Bitboard BOARD_BB;
extern Bitboard SQUARE_BB[SquareNum];
extern Bitboard FILE_BB[FileNum];
extern Bitboard RANK_BB[RankNum];
extern Bitboard BETWEEN_BB[SquareNum][SquareNum];

std::ostream& operator << (std::ostream& os, const Bitboard& rhs);

#endif // !BITBOARD_HPP