#include"move.hpp"
#include"network.hpp"

int32_t Move::toLabel() const {
#ifdef SMALL_OUTPUT
    Color c = pieceToColor(subject());
    Square to_sq = (c == BLACK ? to() : InvSquare[to()]);
    int32_t to_num = SquareToNum[to_sq];
    int32_t piece_num = (c == BLACK ? pieceToIndex[subject()] - 1
        : -pieceToIndex[subject()] - 1); //EMPTY‚É‚Í‚È‚ç‚È‚¢‚Ì‚Åˆê‚Â‹l‚ß‚é
    //printf("piece_num = %3d, to_num = %3d ", piece_num, to_num);
    //print();
    return piece_num * 81 + to_num;
#else
    return SquareToNum[to()];
#endif
}