#ifndef MOVE_HPP
#define MOVE_HPP

#include"square.hpp"
#include"piece.hpp"
#include"types.hpp"
#include<unordered_map>
#include<iostream>

class Move {
public:
    static constexpr int32_t TURN_BIT = 10;

	//コンストラクタ
    Move() = default;
    Move(int32_t x) : move(x), score(Score(0)) {}
    Move(Square to) : move(to), score(Score(0)) {}
    Move(Square to, Color c) : move(c == BLACK ? to : to | (1 << TURN_BIT)), score(Score(0)) {}

	//表示
	void print() const {
        std::cout << to() << std::endl;
	}

	void printWithScore() const {
        std::cout << to() << "\tscore:" << score << std::endl;
	}
	
	//要素を取り出す関数ら
	inline Square to() const { return Square(move & ~(1 << TURN_BIT)); }
    inline Color color() const { return (move & (1 << TURN_BIT) ? WHITE : BLACK); }

	//演算子オーバーロード
	bool operator==(const Move &rhs) const { return (move == rhs.move); }
	bool operator!=(const Move &rhs) const { return !(*this == rhs); }
	bool operator<(const Move &rhs) const { return (score < rhs.score); }
	bool operator>(const Move &rhs) const { return (score > rhs.score); }

    int32_t toLabel() const {
        return (color() == BLACK ? SquareToNum[to()] : SquareToNum[InvSquare[to()]]);
    }

	//探索時にSeacherクラスから気軽にアクセスできるようpublicにおいてるけど
	int32_t move;
	Score score;
};

//比較用
const Move NULL_MOVE(0);

//sfen形式で出力するオーバーロード
inline std::ostream& operator<<(std::ostream& os, Move m) {
    if (m == NULL_MOVE) {
        os << "PA";
    } else {
        os << fileToString[SquareToFile[m.to()]] << SquareToRank[m.to()];
    }
    return os;
}

//これコンストラクタとかで書いた方がいい気がするけどうまく書き直せなかった
//まぁ動けばいいのかなぁ
static Move stringToMove(std::string input) {
    if (input == "PA") {
        return NULL_MOVE;
    }
    Square to = FRToSquare[File8 - (input[0] - 'A')][input[1] - '0'];
    return Move(to);
}

#endif
