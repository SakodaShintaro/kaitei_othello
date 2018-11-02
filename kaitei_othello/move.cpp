#include"move.hpp"
#include"network.hpp"

int32_t Move::toLabel() const {
    return (color() == BLACK ? SquareToNum[to()] : SquareToNum[InvSquare[to()]]);
}