#ifndef POSITION_HPP
#define POSITION_HPP

#include"square.hpp"
#include"piece.hpp"
#include"move.hpp"
#include"eval_elements.hpp"
#include"types.hpp"
#include"bitboard.hpp"
#include"eval_params.hpp"
#include<random>
#include<cstdint>
#include<unordered_map>

constexpr int MAX_MOVE_LIST_SIZE = 593;

class Position {
public:
    //コンストラクタ
    Position(const EvalParams<DefaultEvalType>& eval_params);

    //初期化
    void init();

    //内部の状態等を表示する関数
    void print() const;
    void printAllMoves() const;
    void printHistory() const;
    void printForDebug() const;

    //一手進める・戻す関数
    void doMove(const Move move);
    void undo();
    void doNullMove();
    void undoNullMove();

    //合法性に関する関数
    bool isLegalMove(const Move move) const;

    //評価値計算
    void calcScoreDiff();
#ifdef USE_NN
    Vec makeOutput() const;
    std::vector<CalcType> policy();
    std::vector<CalcType> maskedPolicy();
    CalcType valueScore();
    CalcType valueScoreForTurn();
#ifdef USE_CATEGORICAL
    std::array<CalcType, BIN_SIZE> valueDist();
#endif
#endif

    //特徴量作成
#ifdef USE_NN
    Features makeFeatures() const;
#else
#endif

    //合法手生成
    std::vector<Move> generateAllMoves() const;

    //ハッシュ
    static void initHashSeed();

    //getter
    Move lastMove() const { return (kifu_.empty() ? NULL_MOVE : kifu_.back()); }
    uint32_t turn_number() const { return turn_number_; }
    Color color() const { return color_; }
    int64_t hash_value() const { return hash_value_; }
    Piece on(const Square sq) const { return board_[sq]; }
    const EvalParams<DefaultEvalType>& evalParams() { return eval_params_; }
private:
    //--------------------
	//    内部メソッド
    //--------------------
    //評価値計算
    void initScore();

    //ハッシュ値の初期化
    void initHashValue();

    //------------------
    //    クラス変数
    //------------------
    //ハッシュの各駒・位置に対する決められた値
    static int64_t HashSeed[PieceNum][SquareNum];

    //------------------------
    //    インスタンス変数
    //------------------------
    //手番
    Color color_;

    //盤面
    Piece board_[SquareNum];

    //盤面の履歴をスタックで管理
    std::vector<std::vector<Piece>> stack_;

    //手数
    uint32_t turn_number_;

    //現局面までの指し手履歴
    std::vector<Move> kifu_;

    //現局面のハッシュ値
    int64_t hash_value_;

    //ハッシュ値の履歴
    std::vector<int64_t> hash_values_;

    //Bitboard類
    Bitboard occupied_all_;
    Bitboard occupied_bb_[ColorNum];

#ifdef USE_NN
    Vec output_;
    bool already_calc_;
#endif

    //評価パラメータへの参照
    const EvalParams<DefaultEvalType>& eval_params_;
};

#endif