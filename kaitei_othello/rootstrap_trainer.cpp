#include"rootstrap_trainer.hpp"
#include"position.hpp"
#include"searcher.hpp"
#include"eval_params.hpp"
#include"thread.hpp"
#include"operate_params.hpp"
#include<iomanip>
#include<experimental/filesystem>
#ifdef _MSC_VER
#include<direct.h>
#elif __GNUC__
#include<sys/stat.h>
#endif

static std::mutex MUTEX;

RootstrapTrainer::RootstrapTrainer(std::string settings_file_path) {
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
        } else if (name == "search_depth") {
            ifs >> SEARCH_DEPTH;
        } else if (name == "optimizer") {
            ifs >> OPTIMIZER_NAME;
            if (!isLegalOptimizer()) {
                std::cerr << "optimizerが不正" << std::endl;
                assert(false);
            }
        } else if (name == "learn_rate") {
            ifs >> LEARN_RATE;
        } else if (name == "momentum_decay") {
            ifs >> MOMENTUM_DECAY;
        } else if (name == "thread_num") {
            ifs >> THREAD_NUM;
            THREAD_NUM = std::min(std::max(1u, THREAD_NUM), std::thread::hardware_concurrency());
            usi_option.thread_num = THREAD_NUM;
        } else if (name == "threshold(0.0~1.0)") {
            ifs >> THRESHOLD;
        } else if (name == "random_move_temperature") {
            ifs >> usi_option.temperature;
        } else if (name == "random_move_num") {
            ifs >> usi_option.random_turn;
        } else if (name == "draw_turn") {
            ifs >> usi_option.draw_turn;
        } else if (name == "draw_score") {
            ifs >> usi_option.draw_score;
        } else if (name == "learn_mode(0or1)") {
            ifs >> LEARN_MODE;
        } else if (name == "deep_coefficient") {
            ifs >> DEEP_COEFFICIENT;
        } else if (name == "decay_rate") {
            ifs >> DECAY_RATE;
        } else if (name == "use_draw_game") {
            ifs >> USE_DRAW_GAME;
        } else if (name == "USI_Hash") {
            ifs >> usi_option.USI_Hash;
        } else if (name == "evaluation_game_num") {
            ifs >> EVALUATION_GAME_NUM;
        } else if (name == "evaluation_interval") {
            ifs >> EVALUATION_INTERVAL;
        } else if (name == "value_coeff") {
            ifs >> VALUE_COEFF;
#ifdef USE_MCTS
        } else if (name == "playout_limit") {
            ifs >> usi_option.playout_limit;
#endif
        }
    }

    //その他オプションを学習用に設定
    shared_data.limit_msec = LLONG_MAX;
    shared_data.stop_signal = false;
    usi_option.byoyomi_margin = 0LL;
    usi_option.resign_score = MIN_SCORE;

    //変数の初期化
    sum_learned_games_ = 0;
    update_num_ = 0;
    fail_num_ = 0;
    consecutive_fail_num_ = 0;
    win_average_ = 0.5;

    //評価関数読み込み
    eval_params->readFile("tmp.bin");
    eval_params->writeFile("before_learn.bin");

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

void RootstrapTrainer::learnAsync() {
    std::cout << "start learnAsync()" << std::endl;
    start_time_ = std::chrono::steady_clock::now();

    //ログファイルの設定
    log_file_.open("learn_async_log.txt");
    print("経過時間");
    print("学習局数");
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

    //スレッドの作成
    std::vector<std::thread> slave_threads(THREAD_NUM);
    for (uint32_t i = 0; i < THREAD_NUM; i++) {
        slave_threads[i] = std::thread(&RootstrapTrainer::learnAsyncSlave, this, i);
    }

    //stopコマンドの入力だけ監視する
    while (true) {
        std::string input;
        std::cin >> input;
        if (input == "stop") {
            shared_data.stop_signal = true;
            break;
        }
    }
    for (uint32_t i = 0; i < THREAD_NUM; i++) {
        slave_threads[i].join();
        printf("%2dスレッドをjoin\n", i);
    }

    log_file_.close();
    std::cout << "finish learnAsync()" << std::endl;
}

void RootstrapTrainer::learnAsyncSlave(int32_t id) {
    //停止信号が来るまでループ
    while(!shared_data.stop_signal) {
        //棋譜を生成
#ifdef USE_MCTS
        auto games = play(BATCH_SIZE, (int32_t)usi_option.playout_limit);
#else
        auto games = play(BATCH_SIZE, SEARCH_DEPTH);
#endif

        //損失・勾配・千日手数・長手数による引き分け数を計算
        std::array<double, 2> loss;
        auto grad = std::make_unique<EvalParams<LearnEvalType>>();
        learnGames(games, loss, *grad);

        MUTEX.lock();
        //パラメータ更新
        updateParams(*eval_params, *grad);

        //tmpファイルとして書き出し
        eval_params->writeFile("tmp.bin");

        //学習局数を更新
        sum_learned_games_ += BATCH_SIZE;

        //学習情報の出力
        timestamp();
        print(sum_learned_games_);
        print(loss[0] + VALUE_COEFF * loss[1]);
        print(loss[0]);
        print(loss[1]);
        print(LEARN_RATE * grad->maxAbs());
        print(LEARN_RATE * grad->sumAbs());
        print(eval_params->maxAbs());
        print(eval_params->sumAbs());

        if (sum_learned_games_ % (BATCH_SIZE * EVALUATION_INTERVAL) == 0) {
            evaluate();
        }
        std::cout << std::endl;
        log_file_ << std::endl;

        MUTEX.unlock();
    }
}

std::vector<Game> RootstrapTrainer::play(int32_t game_num, int32_t search_limit) {
#ifdef USE_MCTS
    auto searcher = std::make_unique<MCTSearcher>(usi_option.USI_Hash);
#else
    auto searcher = std::make_unique<Searcher>(Searcher::SLAVE);
#endif

    std::vector<Game> games(game_num);

    for (int32_t i = 0; i < game_num; i++) {
        Game& game = games[i];
        Position pos(*eval_params);

        while (!pos.isFinish()) {
            //iが偶数のときpos_cが先手
            auto move_and_teacher = searcher->thinkForGenerateLearnData(pos, search_limit);
            Move best_move = move_and_teacher.first;
            TeacherType teacher = move_and_teacher.second;

            if (!pos.isLegalMove(best_move)) {
                pos.printForDebug();
                best_move.printWithScore();
                assert(false);
            }
            pos.doMove(best_move);
            game.moves.push_back(best_move);
            game.teachers.push_back(teacher);
        }

        //対局結果の設定
        int32_t num = pos.score();
        if (num == 0) {
            game.result = 0.5;
        } else if (num > 0) {
            game.result = 1.0;
        } else {
            game.result = 0.0;
        }
    }
    return games;
}

std::vector<Game> RootstrapTrainer::parallelPlay(const EvalParams<DefaultEvalType>& curr, const EvalParams<DefaultEvalType>& target, int32_t game_num, int32_t search_limit) {
    std::vector<Game> games(game_num);
    std::atomic<int32_t> index;
    index = 0;

    std::vector<std::thread> threads;
    for (int32_t i = 0; i < (int32_t)usi_option.thread_num; i++) {
        threads.emplace_back([&]() {
#ifdef USE_MCTS
            auto searcher = std::make_unique<MCTSearcher>(usi_option.USI_Hash);
#else
            auto searcher = std::make_unique<Searcher>(Searcher::SLAVE);
#endif
            while (true) {
                int32_t curr_index = index++;
                if (curr_index >= game_num) {
                    return;
                }
                Game& game = games[curr_index];
                game.moves.reserve(usi_option.draw_turn);
                game.teachers.reserve(usi_option.draw_turn);
                Position pos_c(curr), pos_t(target);

                while (!pos_c.isFinish()) {
                    //iが偶数のときpos_cが先手
                    auto move_and_teacher = ((pos_c.turn_number() % 2) == (curr_index % 2) ?
                        searcher->thinkForGenerateLearnData(pos_c, search_limit) :
                        searcher->thinkForGenerateLearnData(pos_t, search_limit));
                    Move best_move = move_and_teacher.first;
                    TeacherType teacher = move_and_teacher.second;

                    //if (!pos_c.isLegalMove(best_move)) {
                    //    pos_c.printForDebug();
                    //    best_move.printWithScore();
                    //    assert(false);
                    //}
                    pos_c.doMove(best_move);
                    pos_t.doMove(best_move);
                    game.moves.push_back(best_move);
                    game.teachers.push_back(teacher);
                }

                //対局結果の設定
                int32_t num = pos_c.score();
                if (num == 0) {
                    game.result = 0.5;
                } else if (num > 0) {
                    game.result = 1.0;
                } else {
                    game.result = 0.0;
                }
            }
        });
    }
    for (int32_t i = 0; i < (int32_t)usi_option.thread_num; i++) {
        threads[i].join();
    }
    return games;
}

double RootstrapTrainer::calcCurrWinRate(const std::vector<Game>& games) {
    //currは奇数局で先手、偶数局で後手という前提
    assert(games.size() != 0);
    double win_rate = 0.0;
    for (int32_t i = 0; i < games.size(); i++) {
        win_rate += (i % 2 == 0 ? games[i].result : 1.0 - games[i].result);
    }
    return win_rate / games.size();
}

void RootstrapTrainer::evaluate() {
    //対局するパラメータを準備
    if (!opponent_parameters_) {
        opponent_parameters_ = std::make_unique<EvalParams<DefaultEvalType>>();
    }
    opponent_parameters_->readFile();

    //random_turnは小さめにする
    auto copy = usi_option.random_turn;
    usi_option.random_turn = 6;
#ifdef USE_MCTS
    auto test_games = parallelPlay(*eval_params, *opponent_parameters_, EVALUATION_GAME_NUM, (int32_t)usi_option.playout_limit);
#else
    auto test_games = parallelPlay(*eval_params, *opponent_parameters_, EVALUATION_GAME_NUM, SEARCH_DEPTH);
#endif
    usi_option.random_turn = copy;

    //いくつか出力
    test_games[0].writeKifuFile("./test_games/");
    test_games[1].writeKifuFile("./test_games/");
    test_games[3].writeKifuFile("./test_games/");
    test_games[4].writeKifuFile("./test_games/");

    double win_rate = calcCurrWinRate(test_games);
    if (win_rate >= THRESHOLD) {
        eval_params->writeFile();
        update_num_++;
        consecutive_fail_num_ = 0;
    } else {
        fail_num_++;
        consecutive_fail_num_++;
    }
    print(win_rate * 100.0);
    print(update_num_);
    print(fail_num_);
    print(consecutive_fail_num_);
}

void RootstrapTrainer::learnGames(const std::vector<Game>& games, std::array<double, 2>& loss, EvalParams<LearnEvalType>& grad) {
    loss[0] = loss[1] = 0.0;
    grad.clear();

    //引き分けを除く場合があるのでこれはBATCH_SIZEに一致するとは限らない
    int32_t learn_game_num = 0;

    //一つ書き出してみる
    games.front().writeKifuFile("./learn_games/");

    for (const Game& game : games) {
        //学習
        learn_game_num++;
        if (LEARN_MODE == ELMO_LEARN) {
            loss += learnOneGame(game, grad);
        } else if (LEARN_MODE == N_STEP_SARSA) {
            loss += learnOneGameReverse(game, grad);
        } else { //ここには来ないはず
            assert(false);
        }
    }
    (learn_game_num == 0 ? loss : loss /= learn_game_num);
}

std::array<double, 2> RootstrapTrainer::learnOneGame(const Game& game, EvalParams<LearnEvalType>& grad) {
    std::array<double, 2> loss = { 0.0, 0.0 };
    Position pos(*eval_params);
    uint64_t learn_num = 0;

#ifndef USE_NN //これ探索手法の違いじゃね？
    auto searcher = std::make_unique<Searcher>(Searcher::SLAVE);
#endif
    for (int32_t i = 0; i < game.moves.size(); i++) {
        Move m = game.moves[i];
        if (m == NULL_MOVE) {
            pos.doMove(m);
            continue;
        }

        //学習
        learn_num++;
        TeacherType teacher = game.teachers[i];
        //対局結果を用いてvalueを加工する
#ifdef USE_CATEGORICAL
        double result_for_turn = (pos.color() == BLACK ? game.result : 1.0 - game.result);
        auto dist_for_turn = onehotDist(result_for_turn);
        for (int32_t i = 0; i < BIN_SIZE; i++) {
            teacher[POLICY_DIM + i] =
                (CalcType)(DEEP_COEFFICIENT * teacher[POLICY_DIM + i] +
                    (1.0 - DEEP_COEFFICIENT) * dist_for_turn[i]);
        }
#else
        double deep_win_rate = teacher[POLICY_DIM];
        double result_for_turn = (pos.color() == BLACK ? game.result : 1.0 - game.result);
        double teacher_signal = DEEP_COEFFICIENT * deep_win_rate + (1 - DEEP_COEFFICIENT) * result_for_turn;
        teacher[POLICY_DIM] = (CalcType)teacher_signal;
#endif

        loss += addGrad(grad, pos, teacher);
        if (!pos.isLegalMove(m)) {
            pos.printForDebug();
            m.printWithScore();
        }

        pos.doMove(m);
    }

    return (learn_num == 0 ? loss : loss / learn_num);
}

std::array<double, 2> RootstrapTrainer::learnOneGameReverse(const Game& game, EvalParams<LearnEvalType>& grad) {
    std::array<double, 2> loss = { 0.0, 0.0 };
    auto searcher = std::make_unique<Searcher>(Searcher::SLAVE);
    Position pos(*eval_params);
    uint64_t learn_num = 0;

    //まずは最終局面まで動かす
    for (auto move : game.moves) {
        pos.doMove(move);
    }

    assert(Game::RESULT_WHITE_WIN <= game.result && game.result <= Game::RESULT_BLACK_WIN);

    //先手から見た勝率,分布.指数移動平均で動かしていく.最初は結果によって初期化(0 or 0.5 or 1)
    double win_rate_for_black = game.result;
#ifdef USE_CATEGORICAL
    auto dist_for_black = onehotDist(win_rate_for_black);
#endif

    //最後の局面(詰みや256手,千日手)では教師となる指し手がないので1手戻したところから学習開始
    pos.undo();

    for (size_t i = game.moves.size() - 1; i >= 0; i--) {
        if (game.moves[i].score == MIN_SCORE) { //ランダムムーブということなので学習はしない
            //ランダムムーブは1局の最初の方に行っているのでもう学習終了
            break;
        }

        if (isMatedScore(game.moves[i].score)) { //詰みの値だったら学習を飛ばす
            pos.undo();
            continue;
        }

        //先手から見た勝率について指数移動平均を取り,教師データにセットする
        //教師データをコピーする gameをconstで受け取ってしまっているので
        TeacherType teacher = game.teachers[i];

#ifdef USE_CATEGORICAL
        //teacherから分布を得る
        std::array<CalcType, BIN_SIZE> curr_dist;
        for (int32_t j = 0; j < BIN_SIZE; j++) {
            curr_dist[j] = teacher[POLICY_DIM + j];
        }

        //手番を考慮して先手から見た分布にする
        if (pos.color() == WHITE) {
            std::reverse(curr_dist.begin(), curr_dist.end());
        }

        //混合する
        for (int32_t j = 0; j < BIN_SIZE; j++) {
            dist_for_black[j] = (CalcType)(DECAY_RATE * dist_for_black[j] + (1.0 - DECAY_RATE) * curr_dist[j]);
        }

        //teacherにコピーする
        for (int32_t j = 0; j < BIN_SIZE; j++) {
            teacher[POLICY_DIM + j] = dist_for_black[j];
        }

        //手番に合わせて反転する
        if (pos.color() == WHITE) {
            std::reverse(teacher.begin() + POLICY_DIM, teacher.end());
        }
#else
        //先手から見た値を得る
        double curr_win_rate = (pos.color() == BLACK ? teacher[POLICY_DIM] : -teacher[POLICY_DIM]);

        //混合する
        win_rate_for_black = DECAY_RATE * win_rate_for_black + (1.0 - DECAY_RATE) * curr_win_rate;

        //teacherにコピーする
        teacher[POLICY_DIM]  = (CalcType)(pos.color() == BLACK ? win_rate_for_black  : -win_rate_for_black);
#endif
        //損失・勾配の計算
        loss += addGrad(grad, pos, teacher);
        //学習局面数を増やす
        learn_num++;

        //この局面の学習は終わったので1手戻す
        pos.undo();
    }

    return (learn_num == 0 ? loss : loss / learn_num);
}

void RootstrapTrainer::learnSync() {
    //自己対局だけを並列化
    std::cout << "start learnSync()" << std::endl;

    //時間を設定
    start_time_ = std::chrono::steady_clock::now();

    //ログファイルを準備
    log_file_.open("learn_sync_log.txt");
    print("経過時間");
    print("学習局数");
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

    //ここから学習のメイン
    while (true) {
        //自己対局による棋譜生成:並列化
#ifdef USE_MCTS
        auto games = parallelPlay(*eval_params, *eval_params, BATCH_SIZE, (int32_t)usi_option.playout_limit);
#else
        auto games = parallelPlay(*eval_params, *eval_params, BATCH_SIZE, SEARCH_DEPTH);
#endif
        //損失・勾配・千日手数・長手数による引き分け数を計算
        std::array<double, 2> loss;
        auto grad = std::make_unique<EvalParams<LearnEvalType>>();
        learnGames(games, loss, *grad);

        //パラメータ更新
        updateParams(*eval_params, *grad);
        //書き出し
        eval_params->writeFile("tmp.bin");

        //学習局数を更新
        sum_learned_games_ += BATCH_SIZE;

        //学習情報の表示
        timestamp();
        print(sum_learned_games_);
        print(loss[0] + VALUE_COEFF * loss[1]);
        print(loss[0]);
        print(loss[1]);
        print(LEARN_RATE * grad->maxAbs());
        print(LEARN_RATE * grad->sumAbs());
        print(eval_params->maxAbs());
        print(eval_params->sumAbs());

        //評価
        if (sum_learned_games_ % (BATCH_SIZE * EVALUATION_INTERVAL) == 0) {
            evaluate();
        }

        std::cout << std::endl;
        log_file_ << std::endl;
    }

    log_file_.close();
    std::cout << "finish learnAsync()" << std::endl;
}

void RootstrapTrainer::testLearn() {
    std::cout << "start testLearn()" << std::endl;

    //時間を設定
    start_time_ = std::chrono::steady_clock::now();

    //テスト用に設定
    BATCH_SIZE = 1;

    //自己対局による棋譜生成:並列化
#ifdef USE_MCTS
    std::vector<Game> games = parallelPlay(*eval_params, *eval_params, BATCH_SIZE, (int32_t)usi_option.playout_limit);
#else
    std::vector<Game> games = parallelPlay(*eval_params, *eval_params, BATCH_SIZE, SEARCH_DEPTH);
#endif

    {
        Position pos(*eval_params);
        for (auto move : games.front().moves) {
            pos.doMove(move);
            pos.print();
        }
    }

    std::cout << std::fixed;

    //ここから学習のメイン
    std::vector<double> vcs = { 1.0 };
    std::vector<double> lrs = { 0.0001, 0.00001, 0.000001 };
    for (int32_t i = 0; i < vcs.size(); i++) {
        for (int32_t j = 0; j < lrs.size(); j++) {
            VALUE_COEFF = vcs[i];
            LEARN_RATE = lrs[j];
            eval_params->readFile();

            pre_update_->clear();

            std::ofstream ofs("test_learn_log" + std::to_string(i) + "_" + std::to_string(j) + ".txt");
            ofs << "step";
            ofs << "\tVALUE_COEFF = " << VALUE_COEFF << ", LEARN_RATE = " << LEARN_RATE;
            ofs << "\tVALUE_COEFF = " << VALUE_COEFF << ", LEARN_RATE = " << LEARN_RATE << std::endl;

            for (int64_t i = 0; i < 1000; i++) {
                //損失・勾配・千日手数・長手数による引き分け数を計算
                std::array<double, 2> loss;
                auto grad = std::make_unique<EvalParams<LearnEvalType>>();
                learnGames(games, loss, *grad);

                //パラメータ更新
                updateParams(*eval_params, *grad);
                std::cout << i << "\tloss[0] = " << loss[0] << ",\tloss[1] = " << loss[1] << "\t" << loss[1] * VALUE_COEFF << std::endl;
                ofs << i << "\t" << loss[0] << "\t" << loss[1] << std::endl;
            }

//            for (auto game : games) {
//                Position pos(*eval_params);
//
//                for (auto move : game.moves) {
//                    pos.print();
//                    auto policy = pos.maskedPolicy();
//                    std::cout << "policy[" << move << "] = " << policy[move.toLabel()];
//#ifdef USE_CATEGORICAL
//                    std::cout << std::endl;
//#else
//                    std::cout << ", value = " << sigmoid(pos.valueScoreForTurn(), 1.0) << std::endl;
//#endif
//                    pos.doMove(move);
//                }
//            }
        }
    }
    std::cout << "finish testLearn()" << std::endl;
}