#include"searcher.hpp"
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

std::pair<Move, TeacherType> AlphaBetaSearcher::thinkForGenerateLearnData(Position& root, bool add_noise) {
    //思考開始時間をセット
    start_ = std::chrono::steady_clock::now();

    //ルート局面の合法手を設定
    root_moves_ = root.generateAllMoves();

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
        moves[i].score = policy[moves[i].toLabel()];
        pos.undo();
    }

    score = softmax(score, temperature);
    return moves[randomChoise(score)];
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
    // nodeの種類
    bool isRootNode = (distance_from_root == 0);

    //-----------------------------
    // Step1. 初期化
    //-----------------------------

    //assert類
    assert(MIN_SCORE <= alpha && alpha < beta && beta <= MAX_SCORE);

    //探索局面数を増やす
    ++node_number_;

    if (depth < PLY) {
        return pos.valueScoreForTurn();
    }

    //-----------------------------
    // RootNode以外での処理
    //-----------------------------
    if (!isRootNode) {
        //停止確認
        if (!train_mode && shouldStop()) {
            return SCORE_ZERO;
        }
    }

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

        return tt_score;
    }

    //-----------------------------
    // Step5. 局面の静的評価
    //-----------------------------

    //変数の準備
    Score best_score = MIN_SCORE;
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

        //-----------------------------
        // Step14. 1手進める
        //-----------------------------
        pos.doMove(current_move);

        //探索
        Score score = -search<train_mode>(pos, -beta, -alpha, depth - PLY, distance_from_root + 1);

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

            if (score >= beta) {
                //fail-high
                break; //betaカット
            } else if (score > alpha) {
                alpha = score;
            }
        }
    }

    //-----------------------------
    // 置換表に保存
    //-----------------------------
    hash_table_.save(pos.hash_value(), best_move, best_score, depth, moves);

    return best_score;
}