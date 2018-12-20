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
    //value‚Ì‚Æ‚±‚ë‚¾‚¯1.0, ‘¼‚Í0.0‚Æ‚µ‚½•ª•z‚ð•Ô‚·
    //value / (1.0 / BIN_SIZE) = value * BIN_SIZE ‚Ì‚Æ‚±‚ë‚¾‚¯1.0
    //value = 1.0‚¾‚Æ‚¿‚å‚¤‚ÇBIN_SIZE‚É‚È‚Á‚Ä‚µ‚Ü‚¤‚©‚çmin‚ðŽæ‚é
    std::array<CalcType, BIN_SIZE> result{};
    result[valueToIndex(value)] = 1.0f;
    return result;
}

inline double BernoulliDist(double y, double mu) {
    return std::pow(mu, y) * std::pow(1.0 - mu, 1.0 - y);
}

#endif

#endif