#pragma once

#ifndef OPERATE_PARAMS_HPP
#define OPERATE_PARAMS_HPP

#include"eval_params.hpp"

#ifdef USE_CATEGORICAL
inline std::array<CalcType, BIN_SIZE> onehotDist(double value) {
    //valueのところだけ1.0, 他は0.0とした分布を返す
    //value / (1.0 / BIN_SIZE) = value * BIN_SIZE のところだけ1.0
    //value = 1.0だとちょうどBIN_SIZEになってしまうからminを取る
    int32_t index = std::min((int32_t)(value * BIN_SIZE), BIN_SIZE - 1);
    std::array<CalcType, BIN_SIZE> result;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        result[i] = (CalcType)(i == index ? 1.0 : 0.0);
    }
    return result;
}

inline double BernoulliDist(double y, double mu) {
    return std::pow(mu, y) * std::pow(1.0 - mu, 1.0 - y);
}

#endif

#endif