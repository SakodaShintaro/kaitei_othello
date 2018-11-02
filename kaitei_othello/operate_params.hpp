#pragma once

#ifndef OPERATE_PARAMS_HPP
#define OPERATE_PARAMS_HPP

#include"eval_params.hpp"

#ifdef USE_CATEGORICAL
inline std::array<CalcType, BIN_SIZE> onehotDist(double value) {
    //valueForBlack�̂Ƃ��낾��1.0, ����0.0�Ƃ������z��Ԃ�
    //valueForBlack / (1.0 / BIN_SIZE) = valueForBlack * BIN_SIZE �̂Ƃ��낾��1.0
    //round�����������������
    //valueForBlack = 1.0���Ƃ��傤��BIN_SIZE�ɂȂ��Ă��܂�����min�����
    int32_t index = std::min((int32_t)round(value * BIN_SIZE), BIN_SIZE - 1);
    std::array<CalcType, BIN_SIZE> result;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        result[i] = (CalcType)(i == index ? 1.0 : 0.0);
    }
    return result;
}

//���z�ɂ����Ď�Ԃ𔽓]<=>�z��̏����𔽓]
inline std::array<CalcType, BIN_SIZE> reverseDist(std::array<CalcType, BIN_SIZE> dist) {
    std::reverse(dist.begin(), dist.end());
    return dist;
}

#endif

#endif