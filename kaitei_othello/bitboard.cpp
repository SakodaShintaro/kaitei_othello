#include"bitboard.hpp"

Bitboard BOARD_BB;
Bitboard SQUARE_BB[SquareNum];
Bitboard FILE_BB[FileNum];
Bitboard RANK_BB[RankNum];
Bitboard PROMOTION_ZONE_BB[ColorNum];
Bitboard FRONT_BB[ColorNum][RankNum];
Bitboard BETWEEN_BB[SquareNum][SquareNum];
Bitboard ADJACENT_CONTROL_BB[PieceNum];

Bitboard PAWN_CONTROL_BB[ColorNum][SquareNum];
Bitboard KNIGHT_CONTROL_BB[ColorNum][SquareNum];
Bitboard SILVER_CONTROL_BB[ColorNum][SquareNum];
Bitboard GOLD_CONTROL_BB[ColorNum][SquareNum];

Bitboard BishopEffect[2][SquareNum][128];
Bitboard BishopEffectMask[2][SquareNum];

uint64_t RookFileEffect[RankNum][128];
Bitboard RookRankEffect[FileNum][128];

Bitboard KING_CONTROL_BB[SquareNum];

int Slide[] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1,
    -1, 10, 10, 10, 10, 10, 10, 10, 10, 10, -1,
    -1, 19, 19, 19, 19, 19, 19, 19, 19, 19, -1,
    -1, 28, 28, 28, 28, 28, 28, 28, 28, 28, -1,
    -1, 37, 37, 37, 37, 37, 37, 37, 37, 37, -1,
    -1, 46, 46, 46, 46, 46, 46, 46, 46, 46, -1,
    -1, 55, 55, 55, 55, 55, 55, 55, 55, 55, -1,
    -1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1,
    -1, 10, 10, 10, 10, 10, 10, 10, 10, 10, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
};

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

    //3.PROMOTION_ZONE_BBとFRONT_BB
    PROMOTION_ZONE_BB[BLACK] = RANK_BB[Rank1] | RANK_BB[Rank2] | RANK_BB[Rank3];
    PROMOTION_ZONE_BB[WHITE] = RANK_BB[Rank7] | RANK_BB[Rank8] | RANK_BB[Rank9];

    for (int rank = Rank1; rank <= Rank9; ++rank) {
        for (int black_front = rank - 1; black_front >= Rank1; --black_front) {
            FRONT_BB[BLACK][rank] |= RANK_BB[black_front];
        }
        for (int white_front = rank + 1; white_front <= Rank9; ++white_front) {
            FRONT_BB[WHITE][rank] |= RANK_BB[white_front];
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

    //4.飛び利き
    auto indexToOccupied = [](const int index, const int bits, const Bitboard& mask_) {
        auto mask = mask_;
        auto result = Bitboard(0, 0);
        for (int i = 0; i < bits; ++i) {
            const Square sq = mask.pop();
            if (index & (1 << i))
                result |= SQUARE_BB[sq];
        }
        return result;
    };

    //角の利きのために用意しておく
    //n = 0が右上・左下
    //n = 1が左上・右下
    static const Dir diagonal_deltas[2][2] = {
        { RU, LD },
        { RD, LU }
    };

    auto calcBishopEffectMask = [](Square sq, int n) {
        auto result = Bitboard(0, 0);

        for (auto delta : diagonal_deltas[n]) {
            for (Square to = sq + delta; isOnBoard(to); to = to + delta) {
                result |= SQUARE_BB[to];
            }
        }

        //端は関係ないので外す
        result = result & ~(FILE_BB[File1]);
        result = result & ~(FILE_BB[File9]);
        result = result & ~(RANK_BB[Rank1]);
        result = result & ~(RANK_BB[Rank9]);
        return result;
    };

    //角の利きを初期化
    for (int n : {0, 1}) {
        for (auto sq : SquareList) {
            auto& mask = BishopEffectMask[n][sq];
            mask = calcBishopEffectMask(sq, n);

            assert(!mask.crossOver());

            //全てのBitが立っている場合が最大
            const int bits = static_cast<int>(mask.pop_count());
            const int num = 1 << bits;
            for (int i = 0; i < num; ++i) {
                //邪魔駒の位置を示すindexであるiからoccupiedへ変換する
                Bitboard occupied = indexToOccupied(i, bits, mask);
                uint64_t index = occupiedToIndex(occupied, BishopEffectMask[n][sq]);

                //occupiedを考慮した利きを求める
                for (auto delta : diagonal_deltas[n]) {
                    for (Square to = sq + delta; isOnBoard(to); to = to + delta) {
                        BishopEffect[n][sq][index] |= SQUARE_BB[to];

                        //邪魔駒があったらそこまで
                        if (occupied & SQUARE_BB[to])
                            break;
                    }
                }
            }
        }
    }

    //飛車の縦方向
    for (int rank = Rank1; rank <= Rank9; ++rank) {
        const int num1s = 7;
        for (int i = 0; i < (1 << num1s); ++i) {
            //iが邪魔駒の配置を表したindex
            //1つシフトすればそのまま2~8段目のマスの邪魔駒を表す
            int occupied = i << 1;
            uint64_t bb = 0;
            
            //上に利きを伸ばす
            for (int r = rank - 1; r >= Rank1; --r) {
                bb |= (1LL << SquareToNum[FRToSquare[File1][r]]);
                //邪魔駒があったらそこまで
                if (occupied & (1 << (r - Rank1)))
                    break;
            }

            //下に利きを伸ばす
            for (int r = rank + 1; r <= Rank9; ++r) {
                bb |= (1LL << SquareToNum[FRToSquare[File1][r]]);
                //邪魔駒があったらそこまで
                if (occupied & (1 << (r - Rank1)))
                    break;
            }
            RookFileEffect[rank][i] = bb;
        }
    }
}