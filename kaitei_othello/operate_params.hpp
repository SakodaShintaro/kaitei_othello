#pragma once

#ifndef OPERATE_PARAMS_HPP
#define OPERATE_PARAMS_HPP

#include"eval_params.hpp"

#ifdef USE_CATEGORICAL
inline std::array<CalcType, BIN_SIZE> onehotDist(double value) {
    //valueForBlack‚Ì‚Æ‚±‚ë‚¾‚¯1.0, ‘¼‚Í0.0‚Æ‚µ‚½•ª•z‚ğ•Ô‚·
    //valueForBlack / (1.0 / BIN_SIZE) = valueForBlack * BIN_SIZE ‚Ì‚Æ‚±‚ë‚¾‚¯1.0
    //roundæ‚Á‚½•û‚ª‚¢‚¢‚©‚È
    //valueForBlack = 1.0‚¾‚Æ‚¿‚å‚¤‚ÇBIN_SIZE‚É‚È‚Á‚Ä‚µ‚Ü‚¤‚©‚çmin‚ğæ‚é
    int32_t index = std::min((int32_t)round(value * BIN_SIZE), BIN_SIZE - 1);
    std::array<CalcType, BIN_SIZE> result;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        result[i] = (CalcType)(i == index ? 1.0 : 0.0);
    }
    return result;
}

//•ª•z‚É‚¨‚¢‚Äè”Ô‚ğ”½“]<=>”z—ñ‚Ì‡˜‚ğ”½“]
inline std::array<CalcType, BIN_SIZE> reverseDist(std::array<CalcType, BIN_SIZE> dist) {
    std::reverse(dist.begin(), dist.end());
    return dist;
}

#endif

#endif