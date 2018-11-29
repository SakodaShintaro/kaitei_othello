#pragma once

#ifndef OPERATE_PARAMS_HPP
#define OPERATE_PARAMS_HPP

#include"eval_params.hpp"

#ifdef USE_CATEGORICAL
inline std::array<CalcType, BIN_SIZE> onehotDist(double value) {
    //valueForBlack�̂Ƃ��낾��1.0, ����0.0�Ƃ������z��Ԃ�
    //valueForBlack / (1.0 / BIN_SIZE) = valueForBlack * BIN_SIZE �̂Ƃ��낾��1.0
    //valueForBlack = 1.0���Ƃ��傤��BIN_SIZE�ɂȂ��Ă��܂�����min�����
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