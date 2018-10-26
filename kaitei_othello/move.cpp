#include"move.hpp"
#include"network.hpp"

int32_t Move::toLabel() const {
#ifdef SMALL_OUTPUT
    Color c = pieceToColor(subject());
    Square to_sq = (c == BLACK ? to() : InvSquare[to()]);
    int32_t to_num = SquareToNum[to_sq];
    int32_t piece_num = (c == BLACK ? pieceToIndex[subject()] - 1
        : -pieceToIndex[subject()] - 1); //EMPTY�ɂ͂Ȃ�Ȃ��̂ň�l�߂�
    //printf("piece_num = %3d, to_num = %3d ", piece_num, to_num);
    //print();
    return piece_num * 81 + to_num;
#else
    Color c = pieceToColor(subject());
    Square to_sq = (c == BLACK ? to() : InvSquare[to()]);
    Square from_sq = (c == BLACK ? from() : InvSquare[from()]);

    //�ړ���̃}�X
    int32_t to_num = SquareToNum[to_sq];

    //�ړ�������̕���
    int32_t direction;
    File to_file = SquareToFile[to_sq];
    Rank to_rank = SquareToRank[to_sq];
    File from_file = SquareToFile[from_sq];
    Rank from_rank = SquareToRank[from_sq];

    if (from() == WALL00) { //�ł�
        direction = 20 + kind(subject()) - PAWN;
    } else if (to_file == from_file - 1 && to_rank == from_rank + 2) { //�j�n
        direction = 4;
    } else if (to_file == from_file + 1 && to_rank == from_rank + 2) { //�j�n
        direction = 6;
    } else if (to_file == from_file && to_rank > from_rank) { //��
        direction = 0;
    } else if (to_file > from_file && to_rank > from_rank) { //�E��
        direction = 1;
    } else if (to_file > from_file && to_rank == from_rank) { //�E
        direction = 2;
    } else if (to_file > from_file && to_rank < from_rank) { //�E��
        direction = 3;
    } else if (to_file == from_file && to_rank < from_rank) { //��
        direction = 5;
    } else if (to_file < from_file && to_rank < from_rank) { //����
        direction = 7;
    } else if (to_file < from_file && to_rank == from_rank) { //��
        direction = 8;
    } else if (to_file < from_file && to_rank > from_rank) { //����
        direction = 9;
    }
    if (isPromote()) {
        direction += 10;
    }

    return to_num * 27 + direction;
#endif
}
