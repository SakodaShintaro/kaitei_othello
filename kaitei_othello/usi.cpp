#include"usi.hpp"
#include"move.hpp"
#include"position.hpp"
#include"searcher.hpp"
#include"usi_options.hpp"
#include"shared_data.hpp"
#include"eval_params.hpp"
#include"game.hpp"
#include"rootstrap_trainer.hpp"
#include"treestrap_trainer.hpp"
#include"alhpazero_trainer.hpp"
#include"test.hpp"
#include"network.hpp"
#include"MCTSearcher.hpp"
#include"operate_params.hpp"
#include<iostream>
#include<string>
#include<thread>

USIOption usi_option;

void NBoardProtocol::loop() {
    std::string input;
    std::ofstream log("log.txt");

    //設定してしまう
    usi_option.USI_Hash = 256;
    usi_option.byoyomi_margin = 0;
    usi_option.random_turn = 0;
    usi_option.thread_num = 1;
    usi_option.draw_score = -1;
    usi_option.draw_turn = 256;
    usi_option.temperature = 10.0;
    usi_option.resign_score = MIN_SCORE;
    usi_option.playout_limit = 80000;
    shared_data.limit_msec = 10000;

    //これはαβ探索用なので当面は使わない
    shared_data.hash_table.setSize(1);

    //評価関数のロード
    eval_params->readFile();

    //探索クラスの準備
    MCTSearcher mctsearcher(usi_option.USI_Hash);

    while (true) {
        std::cin >> input;
        log << input << std::endl;
        if (input == "go") {
            shared_data.stop_signal = false;
#ifdef USE_MCTS
            mctsearcher.think();
#endif
        } else if (input == "prepareForLearn") {
            eval_params->initRandom();
            eval_params->printHistgram();
            eval_params->writeFile();
            eval_params->writeFile("tmp.bin");
            std::cout << "0初期化したパラメータを出力" << std::endl;
        } else if (input == "learnAsync") {
            RootstrapTrainer trainer("rootstrap_settings.txt");
            trainer.learnAsync();
        } else if (input == "learnSync") {
            RootstrapTrainer trainer("rootstrap_settings.txt");
            trainer.learnSync();
        } else if (input == "treeStrap") {
            TreestrapTrainer trainer("treestrap_settings.txt");
            trainer.startLearn();
        } else if (input == "alphaZero") {
            AlphaZeroTrainer trainer("alphazero_settings.txt");
            trainer.learn();
        } else if (input == "printEvalParams") {
            eval_params->readFile();
            eval_params->printHistgram();
        } else if (input == "testMakeRandomPosition") {
            testMakeRandomPosition();
#ifdef USE_CATEGORICAL
        } else if (input == "testOneHotDist") {
            testOneHotDist();
#endif
        } else if (input == "testNN") {
            testNN();
        } else if (input == "testLearn") {
            RootstrapTrainer trainer("rootstrap_settings.txt");
            trainer.testLearn();
        } else if (input == "vsHuman") {
            vsHuman();
        } else if (input == "vsAI") {
            vsAI();
        } else if (input == "nboard") {
            int32_t version;
            std::cin >> version;
            log << version << std::endl;
            //さて何かやることはあるだろうか
            std::cout << "set myname kaitei_othello" << std::endl;
        } else if (input == "set") {
            std::string command;
            std::cin >> command;
            log << command << std::endl;
            if (command == "depth") {
                int32_t max_depth;
                std::cin >> max_depth;
                log << max_depth << std::endl;
                //やることは特になし
            } else if (command == "game") {
                //盤面を初期化
                shared_data.root.init();

                while (true) {
                    std::string ggf_str;
                    std::cin >> ggf_str;
                    log << ggf_str << std::endl;
                    if (ggf_str.front() == '*') {
                        //Moveをパースする
                        for (int32_t i = 0; i < ggf_str.size(); i++) {
                            if (ggf_str[i] == 'B' || ggf_str[i] == 'W') {
                                //[から2文字を取ってMoveとして解釈し]まで飛ばす
                                std::string s = ggf_str.substr(i + 2, 2);
                                Move move = stringToMove(s);
                                shared_data.root.doMove(move);
                                while (ggf_str[i] != ']') {
                                    i++;
                                }
                            }
                        }
                    }
                    if (ggf_str.substr(ggf_str.size() - 2) == ";)") {
                        break;
                    }
                }
            } else if (command == "contempt") {
                int32_t contempt;
                std::cin >> contempt;
                log << contempt << std::endl;
                //意味があまりよくわからない
            }
        } else if (input == "move") {
            std::string move_str;
            std::cin >> move_str;
            log << move_str << std::endl;
            //変換
            Move move = stringToMove(move_str);
            shared_data.root.doMove(move);
        } else if (input == "hint") {
            int32_t n;
            std::cin >> n;
            log << n << std::endl;
        } else if (input == "ping") {
            int32_t ping;
            std::cin >> ping;
            log << ping << std::endl;
            std::cout << "pong " << ping << std::endl;
        } else {
            std::cerr << "Illegal input" << std::endl;
        }
    }
}

void NBoardProtocol::vsHuman() {
    Position pos(*eval_params);

    shared_data.stop_signal = false;
    shared_data.limit_msec = LLONG_MAX;
#ifdef USE_MCTS
    MCTSearcher mctsearcher(usi_option.USI_Hash);
#endif

    int32_t human_turn;
    std::cout << "人間が先手なら0,後手なら1を入力: ";
    std::cin >> human_turn;

    while (true) {
        pos.print();
        if (pos.isFinish()) {
            int32_t num = pos.scoreForBlack();
            if (num == 0) {
                printf("引き分け\n");
            } else if (num > 0) {
                printf("先手が%d差で勝ち\n", num);
            } else {
                printf("後手が%d差で勝ち\n", -num);
            }
            break;
        }

        if (pos.generateAllMoves().size() == 0) {
            //数を数える
            break;
        }

        if (pos.turn_number() % 2 == human_turn) {
            while (true) {
                std::cout << "指し手を入力: ";
                std::string move_str;
                std::cin >> move_str;
                if (move_str == "undo") {
                    pos.undo();
                    pos.undo();
                    break;
                } else if (move_str == "PA") {
                    pos.doMove(NULL_MOVE);
                    break;
                } else if (move_str.size() == 2 && 'A' <= move_str[0] && move_str[0] <= 'H'
                    && '1' <= move_str[1] && move_str[1] <= '8') {
                    Move move = stringToMove(move_str);
                    if (pos.isLegalMove(move)) {
                        pos.doMove(move);
                        break;
                    } else {
                        std::cout << "非合法手" << std::endl;
                    }
                } else {
                    std::cout << "不正な入力" << std::endl;
                }
            }
        } else {
            auto result = mctsearcher.thinkForGenerateLearnData(pos, (int32_t)usi_option.playout_limit, false);
            pos.doMove(result.first);
        }
    }
}

void NBoardProtocol::vsAI() {
    std::cout << "対局数: ";
    int64_t game_num;
    std::cin >> game_num;

    std::cout << "1局目の先後(0 or 1): ";
    int64_t turn;
    std::cin >> turn;

    //ランダム手を増やす
    usi_option.random_turn = 10;

    shared_data.stop_signal = false;
    shared_data.limit_msec = LLONG_MAX;
#ifdef USE_MCTS
    MCTSearcher mctsearcher(usi_option.USI_Hash);
#endif

    Position pos(*eval_params);

    const std::string file_name = "move.txt";

    for (int64_t i = 0; i < game_num; i++) {
        pos.init();
        Game game;
        while (!pos.isFinish()) {
            Move move;
            if ((pos.turn_number() + i) % 2 == turn) {
                //思考する
                auto result = mctsearcher.thinkForGenerateLearnData(pos, (int32_t)usi_option.playout_limit, false);
                move = result.first;

                //ファイルに手を書き出す
                std::ofstream ofs(file_name);
                ofs << move << std::endl;
            } else {
                //相手の番
                //正しい手が書き込まれるまで待つ
                while (true) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));

                    std::ifstream ifs(file_name);
                    std::string move_str;
                    ifs >> move_str;
                    move = stringToMove(move_str);
                    if (pos.isLegalMove(move)) {
                        break;
                    }
                    std::cerr << "不正な入力: " << move_str << std::endl;
                }
            }
            pos.doMove(move);
            game.moves.push_back(move);
        }
        game.writeKifuFile("./");
    }
}