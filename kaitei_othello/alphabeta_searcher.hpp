﻿#ifndef ALPHABETA_SEARCHER_HPP
#define ALPHABETA_SEARCHER_HPP

#include"position.hpp"
#include"move.hpp"
#include"hash_table.hpp"
#include"usi_options.hpp"
#include"shared_data.hpp"
#include"search_stack.hpp"
#include"pv_table.hpp"
#include"history.hpp"
#include<chrono>

class AlphaBetaSearcher {
public:
    //コンストラクタ
	AlphaBetaSearcher(int64_t hash_size) : hash_table_(hash_size) {}

    //学習データを生成する関数
    std::pair<Move, TeacherType> think(Position &root, bool add_noise);

    //探索で再帰的に用いる通常の関数
    template<bool train_mode>
    Score search(Position &pos, Score alpha, Score beta, Depth depth, int distance_from_root);

    //局面の合法手から完全ランダムに選択する関数
    static Move randomChoice(Position& pos);

    //局面の合法手を何らかの方法で点数付けしてsoftmax関数
    //をかけた値を基にランダムに選択する関数
    Move softmaxChoice(Position& pos, double temperature);
    
private:
    //--------------------
    //    内部メソッド
    //--------------------
    //停止すべきか確認する関数
    inline bool shouldStop();

    //------------------
    //    メンバ変数
    //------------------
    //置換表
    HashTable hash_table_;

    //探索局面数
    int64_t node_number_;

    //時間
    std::chrono::steady_clock::time_point start_;

    //History 
    //History history_;

    //思考する局面における合法手
    std::vector<Move> root_moves_;

    //技巧風のPV_Table
    //PVTable pv_table_;
};

#endif