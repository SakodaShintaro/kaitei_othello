#include"position.hpp"
#include"piece.hpp"
#include"common.hpp"
#include"eval_params.hpp"
#include"usi_options.hpp"
#include"network.hpp"
#include<iostream>
#include<fstream>

//enum {
//    PAWN_VALUE = 100,
//    LANCE_VALUE = 267,
//    KNIGHT_VALUE = 295,
//    SILVER_VALUE = 424,
//    GOLD_VALUE = 510,
//    BISHOP_VALUE = 654,
//    ROOK_VALUE = 738,
//    PAWN_PROMOTE_VALUE = 614,
//    LANCE_PROMOTE_VALUE = 562,
//    KNIGHT_PROMOTE_VALUE = 586,
//    SILVER_PROMOTE_VALUE = 569,
//    BISHOP_PROMOTE_VALUE = 951,
//    ROOK_PROMOTE_VALUE = 1086,
//};
//
//int piece_value[] = {
//    0, static_cast<int>(PAWN_VALUE * 1.05), static_cast<int>(LANCE_VALUE * 1.05), static_cast<int>(KNIGHT_VALUE * 1.05), static_cast<int>(SILVER_VALUE * 1.05),
//    static_cast<int>(GOLD_VALUE * 1.05), static_cast<int>(BISHOP_VALUE * 1.05), static_cast<int>(ROOK_VALUE * 1.05), 0, 0, //0~9
//    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //10~19
//    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //20~29
//    0, 0, 0, PAWN_VALUE, LANCE_VALUE, KNIGHT_VALUE, SILVER_VALUE, GOLD_VALUE, BISHOP_VALUE, ROOK_VALUE, //30~39
//    0, 0, 0, 0, 0, 0, 0, 0, 0, PAWN_PROMOTE_VALUE, //40~49
//    LANCE_PROMOTE_VALUE, KNIGHT_PROMOTE_VALUE,  SILVER_PROMOTE_VALUE, 0, BISHOP_PROMOTE_VALUE, ROOK_PROMOTE_VALUE, 0, 0, 0, 0, //50~59
//    0, 0, 0, 0, 0, -PAWN_VALUE, -LANCE_VALUE, -KNIGHT_VALUE, -SILVER_VALUE, -GOLD_VALUE, //60~69
//    -BISHOP_VALUE, -ROOK_VALUE, 0, 0, 0, 0, 0, 0, 0, 0, //70~79
//    0, -PAWN_PROMOTE_VALUE, -LANCE_PROMOTE_VALUE, -KNIGHT_PROMOTE_VALUE, -SILVER_PROMOTE_VALUE, 0, -BISHOP_PROMOTE_VALUE, -ROOK_PROMOTE_VALUE, 0, 0 //80~89
//};

//Aperyの駒割り
enum {
    PAWN_VALUE = 100 * 9 / 10,
    LANCE_VALUE = 350 * 9 / 10,
    KNIGHT_VALUE = 450 * 9 / 10,
    SILVER_VALUE = 550 * 9 / 10,
    GOLD_VALUE = 600 * 9 / 10,
    BISHOP_VALUE = 950 * 9 / 10,
    ROOK_VALUE = 1100 * 9 / 10,
    PAWN_PROMOTE_VALUE = 600 * 9 / 10,
    LANCE_PROMOTE_VALUE = 600 * 9 / 10,
    KNIGHT_PROMOTE_VALUE = 600 * 9 / 10,
    SILVER_PROMOTE_VALUE = 600 * 9 / 10,
    BISHOP_PROMOTE_VALUE = 1050 * 9 / 10,
    ROOK_PROMOTE_VALUE = 1550 * 9 / 10,
};

int32_t piece_value[] = {
    0, PAWN_VALUE, LANCE_VALUE, KNIGHT_VALUE, SILVER_VALUE, GOLD_VALUE, BISHOP_VALUE, ROOK_VALUE, 0, 0, //0~9
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //10~19
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //20~29
    0, 0, 0, PAWN_VALUE, LANCE_VALUE, KNIGHT_VALUE, SILVER_VALUE, GOLD_VALUE, BISHOP_VALUE, ROOK_VALUE, //30~39
    0, 0, 0, 0, 0, 0, 0, 0, 0, PAWN_PROMOTE_VALUE, //40~49
    LANCE_PROMOTE_VALUE, KNIGHT_PROMOTE_VALUE,  SILVER_PROMOTE_VALUE, 0, BISHOP_PROMOTE_VALUE, ROOK_PROMOTE_VALUE, 0, 0, 0, 0, //50~59
    0, 0, 0, 0, 0, -PAWN_VALUE, -LANCE_VALUE, -KNIGHT_VALUE, -SILVER_VALUE, -GOLD_VALUE, //60~69
    -BISHOP_VALUE, -ROOK_VALUE, 0, 0, 0, 0, 0, 0, 0, 0, //70~79
    0, -PAWN_PROMOTE_VALUE, -LANCE_PROMOTE_VALUE, -KNIGHT_PROMOTE_VALUE, -SILVER_PROMOTE_VALUE, 0, -BISHOP_PROMOTE_VALUE, -ROOK_PROMOTE_VALUE, 0, 0 //80~89
};

#ifdef USE_NN

void Position::initScore() {
    std::vector<CalcType> input = makeFeatures();
    Vec input_vec = Eigen::Map<const Vec>(input.data(), input.size());
    output_ = eval_params_.w[1] * Network::activationFunction(eval_params_.w[0] * input_vec + eval_params_.b[0]) + eval_params_.b[1];
    already_calc_ = true;
}

void Position::calcScoreDiff() {
    initScore();
}

int32_t Position::score() const {
    int32_t result = 0;
    for (Square sq : SquareList) {
        if (board_[sq] == EMPTY) {
            continue;
        }
        result += (board_[sq] == BLACK_PIECE ? 1 : -1);
    }
    return result;
}

Vec Position::makeOutput() const{
    std::vector<CalcType> input = makeFeatures();
    Vec input_vec = Eigen::Map<const Vec>(input.data(), input.size());
    return eval_params_.w[1] * Network::activationFunction(eval_params_.w[0] * input_vec + eval_params_.b[0]) + eval_params_.b[1];
}

std::vector<CalcType> Position::policy() {
    if (!already_calc_) {
        initScore();
    }
    std::vector<CalcType> policy(POLICY_DIM);
    for (int32_t i = 0; i < POLICY_DIM; i++) {
        policy[i] = output_(i);
    }
    return policy;
}

std::vector<CalcType> Position::maskedPolicy() {
    if (!already_calc_) {
        initScore();
    }
    std::vector<CalcType> policy(POLICY_DIM);
    for (int32_t i = 0; i < POLICY_DIM; i++) {
        policy[i] = output_(i);
    }
    const auto moves = generateAllMoves();
    std::vector<CalcType> dist(moves.size()), y(POLICY_DIM, 0.0);
    for (int32_t i = 0; i < moves.size(); i++) {
        dist[i] = policy[moves[i].toLabel()];
    }
    dist = softmax(dist);
    for (int32_t i = 0; i < moves.size(); i++) {
        y[moves[i].toLabel()] = dist[i];
    }

    return y;
}

CalcType Position::valueScore() {
    if (!already_calc_) {
        initScore();
    }
#ifdef USE_CATEGORICAL
    std::vector<CalcType> categorical_distribution(BIN_SIZE);
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        categorical_distribution[i] = output_[POLICY_DIM + i];
    }
    categorical_distribution = softmax(categorical_distribution);
    CalcType value = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        value += (CalcType)(VALUE_WIDTH * (0.5 + i) * categorical_distribution[i]);
    }
    return value;
#else
    return output_[POLICY_DIM];
#endif
}

CalcType Position::valueScoreForTurn() {
    return (color_ == BLACK ? valueScore() : -valueScore());
}

#ifdef USE_CATEGORICAL
std::array<CalcType, BIN_SIZE> Position::valueDist() {
    if (!already_calc_) {
        initScore();
    }
    std::vector<CalcType> categorical_distribution(BIN_SIZE);
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        categorical_distribution[i] = output_[POLICY_DIM + i];
    }
    categorical_distribution = softmax(categorical_distribution);
    std::array<CalcType, BIN_SIZE> result;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        result[i] =  categorical_distribution[i];
    }
    return result;
}
#endif

std::vector<float> Position::makeFeatures() const {
    std::vector<float> features(INPUT_DIM, 0);
    for (auto sq : SquareList) {
        features[SquareToNum[sq]] = (board_[sq] == BLACK_PIECE ? 1.0f : -1.0f);
    }

    features[INPUT_DIM - 1] = (CalcType)color_;
    return features;
}

#endif