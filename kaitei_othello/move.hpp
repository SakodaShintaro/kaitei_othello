#ifndef MOVE_HPP
#define MOVE_HPP

#include"square.hpp"
#include"piece.hpp"
#include"types.hpp"
#include<unordered_map>
#include<iostream>

class Move {
public:
	//コンストラクタ
	//Move() : move(0), score(Score(0)) {}
    Move() = default;
    Move(int32_t x) : move(x), score(Score(x)) {}

	//表示
	void print() const {
        std::cout << to() << std::endl;
	}

	void printWithScore() const {
        std::cout << to() << "\tscore:" << score << std::endl;
	}
	
	//要素を取り出す関数ら
	inline Square to() const { return Square(move); }

	//演算子オーバーロード
	bool operator==(const Move &rhs) const { return (move == rhs.move); }
	bool operator!=(const Move &rhs) const { return !(*this == rhs); }
	bool operator<(const Move &rhs) const { return (score < rhs.score); }
	bool operator>(const Move &rhs) const { return (score > rhs.score); }

    int32_t toLabel() const;

	//探索時にSeacherクラスから気軽にアクセスできるようpublicにおいてるけど
	int32_t move;
	Score score;
};

//比較用
const Move NULL_MOVE(0);

//sfen形式で出力するオーバーロード
inline std::ostream& operator<<(std::ostream& os, Move m) {
    if (m.isDrop()) {
        os << PieceToSfenStr[kind(m.subject())][0] << '*' << static_cast<int>(SquareToFile[m.to()]) << static_cast<char>(SquareToRank[m.to()] + 'a' - 1);
    } else {
        os << static_cast<int>(SquareToFile[m.from()])
           << static_cast<char>(SquareToRank[m.from()] + 'a' - 1)
           << static_cast<int>(SquareToFile[m.to()]) 
           << static_cast<char>(SquareToRank[m.to()] + 'a' - 1);
        if (m.isPromote())
            os << '+';
    }
    return os;
}

//これコンストラクタとかで書いた方がいい気がするけどうまく書き直せなかった
//まぁ動けばいいのかなぁ
static Move stringToMove(std::string input) {
	static std::unordered_map<char, Piece> charToPiece = {
		{ 'P', PAWN },
		{ 'L', LANCE },
		{ 'N', KNIGHT },
		{ 'S', SILVER },
		{ 'G', GOLD },
		{ 'B', BISHOP },
		{ 'R', ROOK },
	};
	if ('A' <= input[0] && input[0] <= 'Z') { //持ち駒を打つ手
		Square to = FRToSquare[input[2] - '0'][input[3] - 'a' + 1];
		return dropMove(to, charToPiece[input[0]]);
	} else { //盤上の駒を動かす手
		Square from = FRToSquare[input[0] - '0'][input[1] - 'a' + 1];
		Square   to = FRToSquare[input[2] - '0'][input[3] - 'a' + 1];
        bool promote = (input.size() == 5 && input[4] == '+');
        return Move(to, from, false, promote, EMPTY, EMPTY);
	}
}

#endif
