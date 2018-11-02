#pragma once
#ifndef EVAL_ELEMENTS_HPP
#define EVAL_ELEMENTS_HPP

#include"piece.hpp"
#include"square.hpp"
#include"eval_params.hpp"
#include<algorithm>
#include<vector>

constexpr int PIECE_STATE_LIST_SIZE = 38;

//評価値の計算に用いる特徴量をまとめたもの
using Features = std::vector<float>;


#endif // !EVAL_ELEMENT_HPP