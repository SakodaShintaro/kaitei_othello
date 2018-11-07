#include"position.hpp"
#include"piece.hpp"
#include"common.hpp"
#include"eval_params.hpp"
#include"usi_options.hpp"
#include"network.hpp"
#include"operate_params.hpp"
#include<iostream>
#include<fstream>

void Position::initPolicyAndValue() {
    std::vector<CalcType> input = makeFeature();
    Vec input_vec = Eigen::Map<const Vec>(input.data(), input.size());
    Vec x[LAYER_NUM], u[LAYER_NUM];
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        x[i] = (i == 0 ? input_vec : Network::activationFunction(u[i - 1]));
        u[i] = eval_params_.w[i] * x[i] + eval_params_.b[i];
    }
    output_ = u[LAYER_NUM - 1];
    already_calc_ = true;
}

int32_t Position::scoreForBlack() const {
    int32_t result = 0;
    for (Square sq : SquareList) {
        if (board_[sq] == EMPTY) {
            continue;
        }
        result += (board_[sq] == BLACK_PIECE ? 1 : -1);
    }
    return result;
}

double Position::resultForBlack() const {
    int32_t s = scoreForBlack();
    return (s > 0 ? 1.0 : (s < 0 ? 0.0 : 0.5));
}

double Position::resultForTurn() const {
    return (color_ == BLACK ? resultForBlack() : 1.0 - resultForBlack());
}

Vec Position::makeOutput() const{
    std::vector<CalcType> input = makeFeature();
    Vec input_vec = Eigen::Map<const Vec>(input.data(), input.size());
    Vec x[LAYER_NUM], u[LAYER_NUM];
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        x[i] = (i == 0 ? input_vec : Network::activationFunction(u[i - 1]));
        u[i] = eval_params_.w[i] * x[i] + eval_params_.b[i];
    }
    return u[LAYER_NUM - 1];
}

std::vector<CalcType> Position::policy() {
    if (!already_calc_) {
        initPolicyAndValue();
    }
    std::vector<CalcType> policy(POLICY_DIM);
    for (int32_t i = 0; i < POLICY_DIM; i++) {
        policy[i] = output_(i);
    }
    return policy;
}

std::vector<CalcType> Position::maskedPolicy() {
    if (!already_calc_) {
        initPolicyAndValue();
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

CalcType Position::valueScoreForTurn() {
    if (!already_calc_) {
        initPolicyAndValue();
    }
#ifdef USE_CATEGORICAL
    //CategoricalでScoreだけを返す方法がわからない
    return (CalcType)valueForTurn();
#else
    return output_[POLICY_DIM];
#endif
}

double Position::valueForBlack() {
    return (color_ == BLACK ? valueForTurn() : 1.0 - valueForTurn());
}

double Position::valueForTurn() {
    if (!already_calc_) {
        initPolicyAndValue();
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
    return standardSigmoid(output_[POLICY_DIM]);
#endif
}

void Position::resetCalc() {
    already_calc_ = false;
}

#ifdef USE_CATEGORICAL
std::array<CalcType, BIN_SIZE> Position::valueDist() {
    if (!already_calc_) {
        initPolicyAndValue();
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

std::vector<float> Position::makeFeature() const {
    std::vector<float> features(INPUT_DIM, 0.0);
    if (color_ == BLACK) {
        for (auto sq : SquareList) {
            if (board_[sq] == EMPTY) {
                continue;
            }
            features[SquareToNum[sq]] = (board_[sq] == BLACK_PIECE ? 1.0f : -1.0f);
        }
    } else {
        for (auto sq : SquareList) {
            if (board_[InvSquare[sq]] == EMPTY) {
                continue;
            }
            features[SquareToNum[sq]] = (board_[InvSquare[sq]] == BLACK_PIECE ? -1.0f : 1.0f);
        }
    }

    return features;
}