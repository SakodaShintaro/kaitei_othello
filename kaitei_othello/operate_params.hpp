#pragma once

#ifndef OPERATE_PARAMS_HPP
#define OPERATE_PARAMS_HPP

#include"eval_params.hpp"

#ifdef USE_CATEGORICAL
inline std::array<CalcType, BIN_SIZE> onehotDist(double value) {
    //valueForBlack‚Ì‚Æ‚±‚ë‚¾‚¯1.0, ‘¼‚Í0.0‚Æ‚µ‚½•ª•z‚ð•Ô‚·
    //valueForBlack / (1.0 / BIN_SIZE) = valueForBlack * BIN_SIZE ‚Ì‚Æ‚±‚ë‚¾‚¯1.0
    //valueForBlack = 1.0‚¾‚Æ‚¿‚å‚¤‚ÇBIN_SIZE‚É‚È‚Á‚Ä‚µ‚Ü‚¤‚©‚çmin‚ðŽæ‚é
    int32_t index = std::min((int32_t)(value * BIN_SIZE - 0.01), BIN_SIZE - 1);
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