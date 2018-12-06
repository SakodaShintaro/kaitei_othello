﻿#include"searcher.hpp"
#include"move.hpp"
#include"usi_options.hpp"
#include"types.hpp"
#include"shared_data.hpp"
#include"network.hpp"
#include"operate_params.hpp"
#include<iostream>
#include<stdio.h>
#include<string>
#include<algorithm>
#include<utility>
#include<functional>
#include<iomanip>

extern USIOption usi_option;

struct SearchLog {
    uint64_t hash_cut_num;
    uint64_t razoring_num;
    uint64_t futility_num;
    uint64_t null_move_num;
    void print() const {
        printf("hash_cut_num  = %llu\n", hash_cut_num);
        printf("razoring_num  = %llu\n", razoring_num);
        printf("futility_num  = %llu\n", futility_num);
        printf("null_move_num = %llu\n", null_move_num);
    }
};
static SearchLog search_log;

std::pair<Move, TeacherType> AlphaBetaSearcher::thinkForGenerateLearnData(Position& root, bool add_noise) {
    //思考開始時間をセット
    start_ = std::chrono::steady_clock::now();

    //History初期化
    history_.clear();

    //PV初期化
    resetPVTable();

    //ルート局面の合法手を設定
    root_moves_ = root.generateAllMoves();

#ifdef USE_SEARCH_STACK
    SearchStack* ss = searchInfoAt(0);
    ss->killers[0] = ss->killers[1] = NULL_MOVE;
#ifndef SF_SEARCH
    ss->can_null_move = true;
#endif
#endif

    //指定された手数まで完全ランダムに指す
    if (root.turn_number() + 1 <= usi_option.random_turn) {
        static std::random_device rd;
        return { root_moves_[rd() % root_moves_.size()], TeacherType() };
    }

    //探索局面数を初期化
    node_number_ = 0;

    //探索
    //反復深化
    static constexpr Score DEFAULT_ASPIRATION_WINDOW_SIZE = Score(256);
    Score aspiration_window_size = DEFAULT_ASPIRATION_WINDOW_SIZE;
    Score best_score, alpha, beta, previous_best_score;

    for (Depth depth = PLY; depth <= usi_option.depth_limit * PLY; depth += PLY) {
        //seldepth_の初期化
        seldepth_ = depth;

        //探索窓の設定
        if (depth <= 4 * PLY) { //深さ4まではASPIRATION_WINDOWを使わずフルで探索する
            alpha = MIN_SCORE;
            beta = MAX_SCORE;
        } else {
            alpha = std::max(previous_best_score - aspiration_window_size, MIN_SCORE);
            beta = std::min(previous_best_score + aspiration_window_size, MAX_SCORE);
        }

        while (!shouldStop()) { //exactな評価値が返ってくるまでウィンドウを広げつつ探索
            //指し手のスコアを最小にしておかないと変になる
            for (auto& root_move : root_moves_) {
                root_move.score = MIN_SCORE;
            }

            best_score = search<false>(root, alpha, beta, depth, 0);

            if (best_score <= alpha) {
                //fail-low
                beta = (alpha + beta) / 2;
                alpha -= aspiration_window_size;
                aspiration_window_size *= 4;
            } else if (best_score >= beta) {
                //fail-high
                alpha = (alpha + beta) / 2;
                beta += aspiration_window_size;
                aspiration_window_size *= 4;
            } else {
                aspiration_window_size = DEFAULT_ASPIRATION_WINDOW_SIZE;
                break;
            }
        }

        //停止確認してダメだったら保存せずループを抜ける
        if (shouldStop()) {
            break;
        }

        //指し手の並び替え
        std::stable_sort(root_moves_.begin(), root_moves_.end(), std::greater<Move>());

        //置換表への保存
        hash_table_.save(root.hash_value(), root_moves_[0], root_moves_[0].score, depth, root_moves_);

        //今回のイテレーションにおけるスコアを記録
        previous_best_score = best_score;

        //PVtableをリセットしていいよな？
        resetPVTable();
    }

    Move best_move = root_moves_.front();
    TeacherType teacher(OUTPUT_DIM, 0.0);
    teacher[best_move.toLabel()] = 1.0;

#ifdef USE_CATEGORICAL
    auto dist = onehotDist(best_score);
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        teacher[POLICY_DIM + i] = dist[i];
    }
#else
    teacher[POLICY_DIM] = best_score;
#endif

    return { root_moves_.front(), teacher };
}

template<bool isPVNode>
Score AlphaBetaSearcher::qsearch(Position &pos, Score alpha, Score beta, Depth depth, int distance_from_root) {
    return pos.valueScoreForTurn();
}

void AlphaBetaSearcher::sendInfo(Depth depth, std::string cp_or_mate, Score score, Bound bound) {
    if (bound != EXACT_BOUND) {
        //lower_boundとか表示する意味がない気がしてきた
        return;
    }

    auto now_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - start_);
    std::cout << "info time " << std::setw(6) << elapsed.count();

    //GUIへ読み途中の情報を返す
    std::cout << " depth " << std::setw(2) << depth / PLY;
    std::cout << " seldepth " << std::setw(2) << seldepth_ / PLY;
    std::cout << " nodes " << std::setw(10) << node_number_;
    std::cout << " score " << std::setw(4) << cp_or_mate << " " << std::setw(6) << score;
    if (bound == UPPER_BOUND) {
        std::cout << " upperbound";
    } else if (bound == LOWER_BOUND) {
        std::cout << " lowerbound";
    }
    int64_t nps = (elapsed.count() == 0 ? 0 : (int64_t)((double)(node_number_) / elapsed.count() * 1000.0));
    std::cout << " nps " << std::setw(10) << nps;
    std::cout << " hashfull " << std::setw(4) << (int)hash_table_.hashfull();
    std::cout << " pv ";
    if (pv_table_.size() == 0) {
        pv_table_.update(root_moves_[0], 0);
    }
    for (auto move : pv_table_) {
        std::cout << move << " ";
    }
    std::cout << std::endl;
}

Move AlphaBetaSearcher::randomChoice(Position & pos) {
    auto moves = pos.generateAllMoves();
    std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
    static std::random_device seed_gen;
    static std::default_random_engine engine(seed_gen());
    return moves[dist(engine)];
}

Move AlphaBetaSearcher::softmaxChoice(Position& pos, double temperature) {
    auto moves = pos.generateAllMoves();
    if (moves.size() == 0) {
        return NULL_MOVE;
    }
    auto policy = pos.policyScore();
    std::vector<double> score(moves.size());
    for (int i = 0; i < moves.size(); i++) {
        pos.doMove(moves[i]);
        //手番から見たスコア
        moves[i].score = (Score)(int)policy[moves[i].toLabel()];
        pos.undo();
    }

    score = softmax(score, temperature);

    //forDebug
    //pos.print();
    //for (int i = 0; i < moves.size(); i++) {
    //    moves[i].printWithScore();
    //    std::cout << "\t" << scores[i] << std::endl;
    //}

    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    static std::random_device seed_gen;
    static std::default_random_engine engine(seed_gen());

    double random_value = dist(engine);
    double sum = 0.0;
    for (int i = 0; i < moves.size(); i++) {
        sum += score[i];
        if (random_value <= sum) {
            return moves[i];
        }
    }
    return moves[moves.size() - 1];
}

inline int AlphaBetaSearcher::futilityMargin(int depth) {
    return 175 * depth / PLY;
    //return PLY / 2 + depth * 2;
}

inline bool AlphaBetaSearcher::shouldStop() {
    //探索深さの制限も加えるべきか?
    auto now_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - start_);
    if ((elapsed.count() >= shared_data.limit_msec)
        || node_number_ > usi_option.node_limit
        || shared_data.stop_signal) {
        //停止信号をオンにする
        shared_data.stop_signal = true;
        return true;
    }
    return false;
}

#define OMIT_PRUNINGS

template<bool train_mode>
Score AlphaBetaSearcher::search(Position &pos, Score alpha, Score beta, Depth depth, int distance_from_root) {
    if (depth < PLY) {
        return qsearch<train_mode>(pos, alpha, beta, Depth(0), distance_from_root);
    }

    // nodeの種類
    bool isRootNode = (distance_from_root == 0);

    //-----------------------------
    // Step1. 初期化
    //-----------------------------

    //assert類
    assert(MIN_SCORE <= alpha && alpha < beta && beta <= MAX_SCORE);

    //探索局面数を増やす
    ++node_number_;

    //-----------------------------
    // RootNode以外での処理
    //-----------------------------

    if (!isRootNode) {

        //-----------------------------
        // Step2. 探索の停止と引き分けの確認
        //-----------------------------

        //停止確認
        if (!train_mode && shouldStop()) {
            return SCORE_ZERO;
        }

        //-----------------------------
        // Step3. Mate distance pruning
        //-----------------------------

        //合ってるのか怪しいぞ
        alpha = std::max(MIN_SCORE + distance_from_root, alpha);
        beta = std::min(MAX_SCORE - distance_from_root + 1, beta);
        if (alpha >= beta) {
            return alpha;
        }
    }

#ifdef USE_SEARCH_STACK
    //-----------------------------
    // SearchStackの初期化
    //-----------------------------

    SearchStack* ss = searchInfoAt(distance_from_root);
    (ss + 1)->killers[0] = (ss + 1)->killers[1] = NULL_MOVE;
    (ss + 1)->can_null_move = true;
#endif

    //-----------------------------
    // Step4. 置換表を見る
    //-----------------------------

    auto hash_entry = hash_table_.find(pos.hash_value());
    if (train_mode) {
        hash_entry = nullptr;
    }
    Score tt_score = hash_entry ? hash_entry->best_move.score : MIN_SCORE;

    //tt_moveの合法性判定はここではなくMovePicker内で確認している
    Move tt_move = (hash_entry ? hash_entry->best_move : NULL_MOVE);
    Depth tt_depth = (hash_entry ? hash_entry->depth : Depth(0));

    //置換表の値による枝刈り
    if (hash_entry
        && tt_depth >= depth
        && tt_score >= beta) {

        //tt_moveがちゃんとしたMoveならこれでHistory更新
        if (tt_move != NULL_MOVE) {
            history_.updateBetaCutMove(tt_move, depth);
        }
        search_log.hash_cut_num++;
        return tt_score;
    }

    //-----------------------------
    // Step5. 局面の静的評価
    //-----------------------------

    //変数の準備
    Move non_cut_moves[600];
    uint32_t non_cut_moves_index = 0;
    int move_count = 0;
    Move pre = (pos.turn_number() > 0 ? pos.lastMove() : NULL_MOVE);
    Score best_score = MIN_SCORE + distance_from_root;
    Move best_move = NULL_MOVE;

    //指し手を生成
    std::vector<Move> moves;
    if (hash_entry && hash_entry->sorted_moves.size() > 0) {
        moves = hash_entry->sorted_moves;
        bool flag = true;
        for (auto move : moves) {
            flag = (flag && pos.isLegalMove(move));
        }
        if (!flag) {
            moves = pos.scoredAndSortedMoves();
        }
    } else {
        moves = pos.scoredAndSortedMoves();
    }

    //-----------------------------
    // Step11. Loop through moves
    //-----------------------------

    //Score base_line = pos.color() == BLACK ? pos.scores() : -pos.scores();
    //if (distance_from_root == 0) {
    //    printf("base_line = %4d\n", base_line);
    //}

    for (Move current_move : moves) {
        //ルートノードでしか参照されない
        std::vector<Move>::iterator root_move_itr;

        if (isRootNode) {
            root_move_itr = std::find(root_moves_.begin(), root_moves_.end(), current_move);
            if (root_move_itr == root_moves_.end()) {
                //root_moves_に存在しなかったらおかしいので次の手へ
                continue;
            }
        }

        ++move_count;

        //-----------------------------
        // Step13. 動かす前での枝刈り
        //-----------------------------


        //-----------------------------
        // Step14. 1手進める
        //-----------------------------

        //合法性判定は必要かどうか
        //今のところ合法手しかこないはずだけど
#if DEBUG
        if (!pos.isLegalMove(current_move)) {
            pos.isLegalMove(current_move);
            current_move.print();
            pos.printForDebug();
            assert(false);
        }
#endif

        pos.doMove(current_move);

        Score score;
        bool shouldSearchFullDepth = true;

        //-----------------------------
        // Step15. Move countに応じてdepthを減らした探索(Late Move Reduction)
        //-----------------------------

        //-----------------------------
        // Step16. Full Depth Search
        //-----------------------------
        //if (distance_from_root == 0 && current_move.to() == SQ23 && current_move.subject() == BLACK_PAWN
        //    && depth == 3 * PLY) {
        //    printf("------------------------------\n");
        //}

        if (shouldSearchFullDepth) {
            //Null Window Searchでalphaを超えそうか確認
            //これ入れた方がいいのかなぁ
            score = -search<train_mode>(pos, -alpha - 1, -alpha, depth - PLY, distance_from_root + 1);

            if (alpha < score && score < beta) {
                //いい感じのスコアだったので再探索
                score = -search<train_mode>(pos, -beta, -alpha, depth - PLY, distance_from_root + 1);
            }
        }

        //for (int32_t i = 0; i < distance_from_root; i++) {
        //    printf("  ");
        //}
        //current_move.scores = scores;
        //current_move.printWithScore();
        //if (distance_from_root == 0 && current_move.to() == SQ23 && current_move.subject() == BLACK_PAWN) {
        //    printf("------------------------------\n");
        //}

        //-----------------------------
        // Step17. 1手戻す
        //-----------------------------
        pos.undo();

        //-----------------------------
        // Step18. 停止確認
        //-----------------------------

        //停止確認
        if (!train_mode && shouldStop()) {
            return Score(0);
        }

        //-----------------------------
        // 探索された値によるalpha更新
        //-----------------------------
        if (score > best_score) {
            if (isRootNode) {
                //ルートノードならスコアを更新しておく
                root_move_itr->score = score;
            }

            best_score = score;
            best_move = current_move;
            pv_table_.update(best_move, distance_from_root);

            if (score >= beta) {
                //fail-high
                break; //betaカット
            } else if (score > alpha) {
                alpha = score;
            }
        }
        non_cut_moves[non_cut_moves_index++] = current_move;
    }


    //-----------------------------
    // Step20. 詰みの確認
    //-----------------------------

    if (move_count == 0) {
        //詰み
        return MIN_SCORE + distance_from_root;
    }

    if (best_move != NULL_MOVE) {
        history_.updateBetaCutMove(best_move, depth);
    }
    for (uint32_t i = 0; i < non_cut_moves_index; i++) {
        history_.updateNonBetaCutMove(non_cut_moves[i], depth);
    }

    //-----------------------------
    // 置換表に保存
    //-----------------------------
    hash_table_.save(pos.hash_value(), best_move, best_score, depth, moves);

    return best_score;
}