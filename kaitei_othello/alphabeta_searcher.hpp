#ifndef ALPHABETA_SEARCHER_HPP
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
    std::pair<Move, TeacherType> thinkForGenerateLearnData(Position &root, bool add_noise);

    //探索で再帰的に用いる通常の関数
    template<bool train_mode>
    Score search(Position &pos, Score alpha, Score beta, Depth depth, int distance_from_root);

    //局面の合法手から完全ランダムに選択する関数
    static Move randomChoice(Position& pos);

    //局面の合法手を何らかの方法で点数付けしてsoftmax関数
    //をかけた値を基にランダムに選択する関数
    Move softmaxChoice(Position& pos, double temperature);
    
    //pvを取り出す関数
    std::vector<Move> pv() {
        std::vector<Move> pv;
        for (Move m : pv_table_) {
            pv.push_back(m);
        }
        return pv;
    }

    //pvのリセットをする関数:BonanzaMethodで呼ばれるためpublicに置いているがprivateにできるかも
    void resetPVTable() {
        pv_table_.reset();
    }

    //historyのリセットをする関数:これもBonanzaMethodで呼ばれる
    //RootStrapでも呼ばれないとおかしいか?
    void clearHistory() {
        history_.clear();
    }

private:
    //--------------------
    //    内部メソッド
    //--------------------
    //静止探索をする関数:searchのtemplateパラメータを変えたものとして実装できないか
    template<bool isPVNode>
    Score qsearch(Position &pos, Score alpha, Score beta, Depth depth, int distance_from_root);

    //GUIへ情報を送る関数
    void sendInfo(Depth depth, std::string cp_or_mate, Score score, Bound bound);

    //停止すべきか確認する関数
    inline bool shouldStop();

    //futilityMarginを計算する関数:razoringのマージンと共通化すると棋力が落ちる.そういうものか
    inline static int futilityMargin(int depth);

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
    History history_;

    //思考する局面における合法手
    std::vector<Move> root_moves_;

    //Seldpth
    Depth seldepth_;

    //技巧風のPV_Table
    PVTable pv_table_;
};

#endif