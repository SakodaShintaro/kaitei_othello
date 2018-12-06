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

#ifndef USE_MCTS

extern USIOption usi_option;

std::pair<Move, TeacherType> AlphaBetaSearcher::thinkForGenerateLearnData(Position& root, bool add_noise) {
    //思考開始時間をセット
    start_ = std::chrono::steady_clock::now();

    //ルート局面の合法手を設定
    root_moves_ = root.generateAllMoves();

    //指定された手数まで完全ランダムに指す
    if (root.turn_number() < usi_option.random_turn) {
        static std::random_device rd;
        return { root_moves_[rd() % root_moves_.size()], TeacherType() };
    }

    //探索局面数を初期化
    node_number_ = 0;

    //反復深化
    for (Depth depth = PLY; depth <= usi_option.depth_limit * PLY; depth += PLY) {
        //指し手のスコアを最小にしておかないと変になる
        for (auto& root_move : root_moves_) {
            root_move.score = MIN_SCORE;
        }

        search<false>(root, MIN_SCORE, MAX_SCORE, depth, 0);

        //停止確認してダメだったら保存せずループを抜ける
        if (shouldStop()) {
            break;
        }

        //指し手の並び替え
        std::stable_sort(root_moves_.begin(), root_moves_.end(), std::greater<Move>());

        //置換表への保存
        hash_table_.save(root.hash_value(), root_moves_[0], root_moves_[0].score, depth, root_moves_);
    }

    TeacherType teacher(OUTPUT_DIM, 0.0);
    teacher[root_moves_[0].toLabel()] = 1.0;

#ifdef USE_CATEGORICAL
    auto dist = onehotDist(root_moves_[0].score);
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        teacher[POLICY_DIM + i] = dist[i];
    }
#else
    teacher[POLICY_DIM] = root_moves_[0].score;
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

template<bool train_mode>
Score AlphaBetaSearcher::search(Position &pos, Score alpha, Score beta, Depth depth, int distance_from_root) {
    // nodeの種類
    const bool isRootNode = (distance_from_root == 0);

    //assert類
    assert(MIN_SCORE <= alpha && alpha < beta && beta <= MAX_SCORE);

    //探索局面数を増やす
    ++node_number_;

    //
    if (pos.isFinish()) {
        return pos.resultForTurn();
    }

    //深さが1未満だったら値を返す
    if (depth < PLY) {
        return pos.valueForTurn();
    }

    //RootNode以外での処理
    if (!isRootNode && !train_mode && shouldStop()) {
        return SCORE_ZERO;
    }

    //置換表の参照
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

    //全探索
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

        //1手進める
        pos.doMove(current_move);

        //探索
        Score score = -search<train_mode>(pos, -beta, -alpha, depth - PLY, distance_from_root + 1);

        //1手戻す
        pos.undo();

        //停止確認
        if (!train_mode && shouldStop()) {
            return Score(0);
        }

        //探索された値によるalpha値の更新
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

#endif