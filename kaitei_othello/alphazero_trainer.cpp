#pragma once

#include"alphazero_trainer.hpp"
#include"rootstrap_trainer.hpp"
#include"position.hpp"
#include"searcher.hpp"
#include"eval_params.hpp"
#include"operate_params.hpp"
#include<iomanip>
#include<algorithm>
#include<experimental/filesystem>
#ifdef _MSC_VER
#include<direct.h>
#elif __GNUC__
#include<sys/stat.h>
#endif

static std::mutex MUTEX;

AlphaZeroTrainer::AlphaZeroTrainer(std::string settings_file_path) {
    //オプションをファイルから読み込む
    std::ifstream ifs(settings_file_path);
    if (!ifs) {
        std::cerr << "fail to open setting_file(" << settings_file_path << ")" << std::endl;
        assert(false);
    }

    std::string name;
    while (ifs >> name) {
        if (name == "batch_size") {
            ifs >> BATCH_SIZE;
        } else if (name == "optimizer") {
            ifs >> OPTIMIZER_NAME;
            if (!isLegalOptimizer()) {
                std::cerr << "optimizerが不正" << std::endl;
                assert(false);
            }
        } else if (name == "learn_rate") {
            ifs >> LEARN_RATE;
        } else if (name == "learn_rate_decay") {
            ifs >> LEARN_RATE_DECAY;
        } else if (name == "momentum_decay") {
            ifs >> MOMENTUM_DECAY;
        } else if (name == "thread_num") {
            ifs >> THREAD_NUM;
            THREAD_NUM = std::min(std::max(1u, THREAD_NUM), std::thread::hardware_concurrency());
            usi_option.thread_num = THREAD_NUM;
        } else if (name == "threshold(0.0~1.0)") {
            ifs >> THRESHOLD;
        } else if (name == "random_move_num") {
            ifs >> usi_option.random_turn;
        } else if (name == "learn_mode(0or1)") {
            ifs >> LEARN_MODE;
        } else if (name == "deep_coefficient") {
            ifs >> DEEP_COEFFICIENT;
        } else if (name == "decay_rate") {
            ifs >> DECAY_RATE;
        } else if (name == "USI_Hash") {
            ifs >> usi_option.USI_Hash;
        } else if (name == "evaluation_game_num") {
            ifs >> EVALUATION_GAME_NUM;
        } else if (name == "evaluation_interval") {
            ifs >> EVALUATION_INTERVAL;
        } else if (name == "evaluation_random_turn") {
            ifs >> EVALUATION_RANDOM_TURN;
        } else if (name == "policy_loss_coeff") {
            ifs >> POLICY_LOSS_COEFF;
        } else if (name == "value_loss_coeff") {
            ifs >> VALUE_LOSS_COEFF;
        } else if (name == "max_stack_size") {
            ifs >> MAX_STACK_SIZE;
            position_stack_.reserve(MAX_STACK_SIZE);
        } else if (name == "max_step_num") {
            ifs >> MAX_STEP_NUM;
#ifdef USE_MCTS
        } else if (name == "playout_limit") {
            ifs >> usi_option.playout_limit;
#endif
        }
    }

    //その他オプションを学習用に設定
    shared_data.limit_msec = LLONG_MAX;
    shared_data.stop_signal = false;

    //Optimizerに合わせて必要なものを準備
    if (OPTIMIZER_NAME == "MOMENTUM") {
        pre_update_ = std::make_unique<EvalParams<LearnEvalType>>();
    }

    //棋譜を保存するディレクトリの削除
    std::experimental::filesystem::remove_all("./learn_games");
    std::experimental::filesystem::remove_all("./test_games");

    //棋譜を保存するディレクトリの作成
#ifdef _MSC_VER
    _mkdir("./learn_games");
    _mkdir("./test_games");
#elif __GNUC__
    mkdir("./learn_games", ACCESSPERMS);
    mkdir("./test_games", ACCESSPERMS);
#endif
}

void AlphaZeroTrainer::learn() {
    std::cout << "start alphaZero()" << std::endl;

    //自己対局スレッドの作成
    std::vector<std::thread> slave_threads(THREAD_NUM - 1);
    for (uint32_t i = 0; i < THREAD_NUM - 1; i++) {
        slave_threads[i] = std::thread(&AlphaZeroTrainer::learnSlave, this);
    }

    //乱数の準備
    std::random_device seed;
    std::default_random_engine engine(seed());

    //局面もインスタンスは一つ用意して都度局面を構成
    Position pos(*eval_params);

    //減衰をかけて壊してしまうので学習率の初期値を保持しておく
    auto start_learning_rate = LEARN_RATE;

    //学習
    for (int32_t i = 1; ; i++) {
        //時間を初期化
        start_time_ = std::chrono::steady_clock::now();

        MUTEX.lock();

        //パラメータの初期化
        eval_params->initRandom();
        eval_params->writeFile("before_learn" + std::to_string(i) + ".bin");

        //変数の初期化
        update_num_ = 0;

        //学習率の初期化
        LEARN_RATE = start_learning_rate;

        //ログファイルの設定
        log_file_.open("alphazero_log" + std::to_string(i) + ".txt");
        print("経過時間");
        print("ステップ数");
        print("損失");
        print("Policy損失");
        print("Value損失");
        print("最大更新量");
        print("総和更新量");
        print("最大パラメータ");
        print("総和パラメータ");
        print("勝率");
        print("勝ち越し数");
        print("負け越し数");
        print("連続負け越し数");
        log_file_ << std::endl << std::fixed;
        std::cout << std::endl << std::fixed;

        //0回目を入れてみる
        timestamp();
        print(0);
        print(0.0);
        print(0.0);
        print(0.0);
        print(0.0);
        print(0.0);
        print(eval_params->maxAbs());
        print(eval_params->sumAbs());
        evaluate();
        std::cout << std::endl;
        log_file_ << std::endl;

        position_stack_.clear();
        position_stack_.reserve(MAX_STACK_SIZE);

        MUTEX.unlock();

        for (int32_t step_num = 1; step_num <= MAX_STEP_NUM; step_num++) {
            //ミニバッチ分勾配を貯める
            auto grad = std::make_unique<EvalParams<LearnEvalType>>();
            std::array<double, 2> loss{ 0.0, 0.0 };
            for (int32_t j = 0; j < BATCH_SIZE; j++) {
                if (position_stack_.size() <= BATCH_SIZE * 10) {
                    j--;
                    continue;
                }

                //ランダムに選ぶ
                MUTEX.lock();
                int32_t random = engine() % position_stack_.size();
                auto data = position_stack_[random];
                MUTEX.unlock();

                //局面を復元
                pos.loadData(data.first);

                //勾配を計算
                loss += addGrad(*grad, pos, data.second);
            }
            loss /= BATCH_SIZE;
            grad->forEach([this](CalcType& g) {
                g /= BATCH_SIZE;
            });

            MUTEX.lock();
            //学習
            updateParams(*eval_params, *grad);

            //学習情報の表示
            timestamp();
            print(step_num);
            print(POLICY_LOSS_COEFF * loss[0] + VALUE_LOSS_COEFF * loss[1]);
            print(loss[0]);
            print(loss[1]);
            print(LEARN_RATE * grad->maxAbs());
            print(LEARN_RATE * grad->sumAbs());
            print(eval_params->maxAbs());
            print(eval_params->sumAbs());

            //学習率の減衰
            LEARN_RATE *= LEARN_RATE_DECAY;

            //評価と書き出し
            if (step_num % EVALUATION_INTERVAL == 0 || step_num == MAX_STEP_NUM) {
                evaluate();
                eval_params->writeFile("tmp" + std::to_string(i) + "_" + std::to_string(step_num) + ".bin");
            }

            std::cout << std::endl;
            log_file_ << std::endl;

            MUTEX.unlock();
        }

        log_file_.close();
    }

    shared_data.stop_signal = true;
    for (uint32_t i = 0; i < THREAD_NUM - 1; i++) {
        slave_threads[i].join();
        printf("%2dスレッドをjoin\n", i);
    }

    log_file_.close();
    std::cout << "finish learnAsync()" << std::endl;
}

void AlphaZeroTrainer::learnSlave() {
    //停止信号が来るまでループ
    while (!shared_data.stop_signal) {
        //棋譜を生成
        auto games = RootstrapTrainer::play(1, true);

        MUTEX.lock();
        //生成した棋譜を学習用データに加工してstackへ詰め込む
        for (auto& game : games) {
            if (LEARN_MODE == ELMO_LEARN) {
                pushOneGame(game);
            } else if (LEARN_MODE == N_STEP_SARSA) {
                pushOneGameReverse(game);
            } else { //ここには来ないはず
                assert(false);
            }
        }

        if ((int64_t)position_stack_.size() >= MAX_STACK_SIZE) {
            auto diff = position_stack_.size() - MAX_STACK_SIZE;
            position_stack_.erase(position_stack_.begin(), position_stack_.begin() + diff);
        }
        MUTEX.unlock();
    }
}

void AlphaZeroTrainer::evaluate() {
    //対局するパラメータを準備
    auto opponent_parameters_ = std::make_unique<EvalParams<DefaultEvalType>>();
    opponent_parameters_->readFile();

    //random_turnは小さめにする
    auto copy = usi_option.random_turn;
    usi_option.random_turn = (uint32_t)EVALUATION_RANDOM_TURN;
    auto test_games = RootstrapTrainer::parallelPlay(*eval_params, *opponent_parameters_, EVALUATION_GAME_NUM, false);
    usi_option.random_turn = copy;

    //出力
    for (int32_t i = 0; i < std::min(10000, (int32_t)test_games.size()); i++) {
        test_games[i].writeKifuFile("./test_games/");
    }

    double win_rate = 0.0;
    for (int32_t i = 0; i < test_games.size(); i++) {
        win_rate += (i % 2 == 0 ? test_games[i].result : 1.0 - test_games[i].result);
    }

    //重複の確認をしてみる
    int32_t same_num = 0;
    for (int32_t i = 0; i < test_games.size(); i++) {
        for (int32_t j = i + 1; j < test_games.size(); j++) {
            if (test_games[i].moves.size() != test_games[i].moves.size()) {
                continue;
            }
            bool same = true;
            for (int32_t k = 0; k < test_games[i].moves.size(); k++) {
                if (test_games[i].moves[k] != test_games[j].moves[k]) {
                    same = false;
                    break;
                }
            }
            if (same) {
                same_num++;
            }
        }
    }
    win_rate /= test_games.size();

    if (win_rate >= THRESHOLD) {
        eval_params->writeFile();
        update_num_++;
    }
    print(win_rate * 100.0);
    print(update_num_);
    print(same_num);
    print(same_num == 0 ? --EVALUATION_RANDOM_TURN : ++EVALUATION_RANDOM_TURN);
}

void AlphaZeroTrainer::pushOneGame(Game& game) {
    Position pos(*eval_params);

    for (int32_t i = 0; i < game.moves.size(); i++) {
        const Move& move = game.moves[i];
        if (move == NULL_MOVE) {
            pos.doMove(move);
            continue;
        }

        //教師信号を計算
        double result_for_turn = (pos.color() == BLACK ? game.result : 1.0 - game.result);
        double teacher_signal = DEEP_COEFFICIENT * game.moves[i].score + (1 - DEEP_COEFFICIENT) * result_for_turn;

#ifdef USE_CATEGORICAL
        //auto dist = pos.valueDist();
        //CalcType sum = 0.0;
        //for (int32_t j = 0; j < BIN_SIZE; j++) {
        //    game.teachers[i][POLICY_DIM + j] = (CalcType)(dist[j] * BernoulliDist(teacher_signal, VALUE_WIDTH * (j + 0.5)));
        //    sum += game.teachers[i][POLICY_DIM + j];
        //}
        //for (int32_t j = 0; j < BIN_SIZE; j++) {
        //    game.teachers[i][POLICY_DIM + j] /= sum;
        //}

        //手番から見た分布を得る
        auto teacher_dist = onehotDist(teacher_signal);

        //teacherにコピーする
        for (int32_t j = 0; j < BIN_SIZE; j++) {
            game.teachers[i][POLICY_DIM + j] = teacher_dist[j];
        }
#else
        game.teachers[i][POLICY_DIM] = (CalcType)teacher_signal;
#endif

        //スタックに詰める
        position_stack_.push_back({ pos.data(), game.teachers[i] });

        pos.doMove(move);
    }
}

void AlphaZeroTrainer::pushOneGameReverse(Game& game) {
    Position pos(*eval_params);

    //まずは最終局面まで動かす
    for (auto move : game.moves) {
        pos.doMove(move);
    }

    assert(Game::RESULT_WHITE_WIN <= game.result && game.result <= Game::RESULT_BLACK_WIN);

    //先手から見た勝率,分布.指数移動平均で動かしていく.最初は結果によって初期化(0 or 0.5 or 1)
    double win_rate_for_black = game.result;

    for (int32_t i = (int32_t)game.moves.size() - 1; i >= 0; i--) {
        //i番目の指し手が対応するのは1手戻した局面
        pos.undo();

        if (game.moves[i] == NULL_MOVE) { 
            //パスだったら学習を飛ばす
            continue;
        }

        //探索結果を先手から見た値に変換
        double curr_win_rate = (pos.color() == BLACK ? game.moves[i].score : 1.0 - game.moves[i].score);

        //混合
        win_rate_for_black = DECAY_RATE * win_rate_for_black + (1.0 - DECAY_RATE) * curr_win_rate;

#ifdef USE_CATEGORICAL
        //手番から見た分布を得る
        auto teacher_dist = onehotDist(pos.color() == BLACK ? win_rate_for_black : 1.0 - win_rate_for_black);

        //teacherにコピーする
        for (int32_t j = 0; j < BIN_SIZE; j++) {
            game.teachers[i][POLICY_DIM + j] = teacher_dist[j];
        }
#else
        //teacherにコピーする
        game.teachers[i][POLICY_DIM] = (CalcType)(pos.color() == BLACK ? win_rate_for_black : 1.0 - win_rate_for_black);
#endif

        //スタックに詰める
        position_stack_.push_back({ pos.data(), game.teachers[i] });
    }
}