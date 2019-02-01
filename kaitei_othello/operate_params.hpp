#pragma once

#ifndef OPERATE_PARAMS_HPP
#define OPERATE_PARAMS_HPP

#include"eval_params.hpp"

#ifdef USE_CATEGORICAL
inline int32_t valueToIndex(double value) {
    assert(0.0 <= value && value <= 1.0);
    return std::min((int32_t)(value * BIN_SIZE), BIN_SIZE - 1);
}

inline std::array<CalcType, BIN_SIZE> onehotDist(double value) {
    //valueのところだけ1.0, 他は0.0とした分布を返す
    //value / (1.0 / BIN_SIZE) = value * BIN_SIZE のところだけ1.0
    //value = 1.0だとちょうどBIN_SIZEになってしまうからminを取る
    std::array<CalcType, BIN_SIZE> result{};
    result[valueToIndex(value)] = 1.0f;
    return result;
}

inline double expOfValueDist(std::array<CalcType, BIN_SIZE> dist) {
    double exp = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        //i番目の要素が示す値は(i + 0.5) * VALUE_WIDTH
        exp += (i + 0.5) * VALUE_WIDTH * dist[i];
    }
    return exp;
}

inline double BernoulliDist(double y, double mu) {
    return std::pow(mu, y) * std::pow(1.0 - mu, 1.0 - y);
}

#endif

#endif