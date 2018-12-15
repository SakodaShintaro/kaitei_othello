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
#include"alphazero_trainer.hpp"
#include"test.hpp"
#include"network.hpp"
#include"MCTSearcher.hpp"
#include"operate_params.hpp"
#include<iostream>
#include<string>
#include<thread>
#ifdef _MSC_VER
#include<Windows.h>
#include<direct.h>
#endif

USIOption usi_option;

void NBoardProtocol::loop() {
    //評価関数のロード
    eval_params->readFile();

    //設定ファイルを読み込む
    std::ifstream ifs("settings.txt");

    if (!ifs) {
        std::cerr << "There is not settings.txt" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(100));
        assert(false);
    }

    std::string input;
    while (ifs >> input) {
        if (input == "hash_size") {
            ifs >> usi_option.USI_Hash;
        } else if (input == "random_turn") {
            ifs >> usi_option.random_turn;
        } else if (input == "thread_num") {
            ifs >> usi_option.thread_num;
#ifdef USE_MCTS
        } else if (input == "playout_limit") {
            ifs >> usi_option.playout_limit;
#else
        } else if (input == "depth_limit") {
            ifs >> usi_option.depth_limit;
        } else if (input == "node_limit") {
            ifs >> usi_option.node_limit;
#endif
        } else if (input == "limit_msec") {
            ifs >> shared_data.limit_msec;
        } else if (input == "eval_params_file") {
            std::string file_name;
            ifs >> file_name;
            eval_params->readFile(file_name);
        }
    }

    //探索クラスの準備
    Searcher searcher(usi_option.USI_Hash);

    while (true) {
        std::cin >> input;
        if (input == "go") {
            shared_data.stop_signal = false;
            auto result = searcher.thinkForGenerateLearnData(root_, false);
            std::cout << "=== " << result.first << std::endl;
        } else if (input == "prepareForLearn") {
            eval_params->initRandom();
            eval_params->printHistgram();
            eval_params->writeFile();
            eval_params->writeFile("tmp.bin");
            std::cout << "0初期化したパラメータを出力" << std::endl;
        } else if (input == "loadFile") {
            std::string file_name;
            std::cout << "読み込む評価関数: ";
            std::cin >> file_name;
            eval_params->readFile(file_name);
        } else if (input == "learnAsync") {
            RootstrapTrainer trainer("rootstrap_settings.txt");
            trainer.learnAsync();
        } else if (input == "learnSync") {
            RootstrapTrainer trainer("rootstrap_settings.txt");
            trainer.learnSync();
#ifndef USE_MCTS
        } else if (input == "treeStrap") {
            TreestrapTrainer trainer("treestrap_settings.txt");
            trainer.startLearn();
#endif
        } else if (input == "alphaZero") {
            AlphaZeroTrainer trainer("alphazero_settings.txt");
            trainer.learn();
        } else if (input == "printEvalParams") {
            eval_params->readFile();
            eval_params->printHistgram();
        } else if (input == "testRandom") {
            testRandom();
#ifdef USE_CATEGORICAL
        } else if (input == "testOneHotDist") {
            testOneHotDist();
        } else if (input == "testDist") {
            testDistEffect();
        } else if (input == "testTreeDist") {
            testTreeDist();
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
            //さて何かやることはあるだろうか
            std::cout << "set myname kaitei_othello" << std::endl;
        } else if (input == "set") {
            std::string command;
            std::cin >> command;
            if (command == "depth") {
                int32_t max_depth;
                std::cin >> max_depth;
                //やることは特になし
            } else if (command == "game") {
                //盤面を初期化
                root_.init();

                while (true) {
                    std::string ggf_str;
                    std::cin >> ggf_str;
                    if (ggf_str.front() == '*') {
                        //Moveをパースする
                        for (int32_t i = 0; i < ggf_str.size(); i++) {
                            if (ggf_str[i] == 'B' || ggf_str[i] == 'W') {
                                //[から2文字を取ってMoveとして解釈し]まで飛ばす
                                std::string s = ggf_str.substr(i + 2, 2);
                                Move move = stringToMove(s);
                                root_.doMove(move);
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
                //意味があまりよくわからない
            }
        } else if (input == "move") {
            std::string move_str;
            std::cin >> move_str;
            //変換
            Move move = stringToMove(move_str);
            root_.doMove(move);
        } else if (input == "hint") {
            int32_t n;
            std::cin >> n;
        } else if (input == "ping") {
            int32_t ping;
            std::cin >> ping;
            std::cout << "pong " << ping << std::endl;
        } else {
            std::cerr << "Illegal input" << std::endl;
        }
    }
}

void NBoardProtocol::vsHuman() {
    Position pos(*eval_params);

    shared_data.stop_signal = false;
    Searcher searcher(usi_option.USI_Hash);

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
            auto result = searcher.thinkForGenerateLearnData(pos, false);
            pos.doMove(result.first);
        }
    }
}

void NBoardProtocol::vsAI() {
    //棋譜を保存するディレクトリの削除
    std::experimental::filesystem::remove_all("./games");

    //棋譜を保存するディレクトリの作成
    _mkdir("./games");

    std::cout << "使用する評価パラメータ: ";
    std::string file_name;
    std::cin >> file_name;
    eval_params->readFile(file_name);

    std::cout << "対局数: ";
    int64_t game_num;
    std::cin >> game_num;

    std::cout << "パイプ名: ";
    std::string pipe_name;
    std::cin >> pipe_name;

    std::cout << "1局目の先後(0 or 1): ";
    int64_t turn;
    std::cin >> turn;

    char sendBuffer[256];
    HANDLE pipe_handle;

    if (turn == 0) {
        pipe_handle = CreateNamedPipe(
            ("\\\\.\\pipe\\mypipe" + pipe_name).c_str(),
            PIPE_ACCESS_DUPLEX, PIPE_TYPE_MESSAGE,
            1,
            sizeof(sendBuffer),
            sizeof(sendBuffer),
            1000, NULL);
        if (pipe_handle == INVALID_HANDLE_VALUE) {
            std::cout << "パイプ作成に失敗" << std::endl;
            return;
        }

        std::cout << "後手の起動待ち..." << std::endl;
        ConnectNamedPipe(pipe_handle, NULL);
    } else {
        pipe_handle = CreateFile(("\\\\.\\pipe\\mypipe" + pipe_name).c_str(),
            GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
        if (pipe_handle == INVALID_HANDLE_VALUE) {
            std::cout << "パイプ作成に失敗" << std::endl;
            return;
        }
    }

    //ランダム手
    usi_option.random_turn = 0;

    shared_data.stop_signal = false;
    Searcher searcher(usi_option.USI_Hash);

    Position pos(*eval_params);

    std::vector<Game> games;
    std::cout << std::fixed;

    double win_point = 0.0;
    for (int64_t i = 0; i < game_num; i++) {
        pos.init();
        Game game;
        while (!pos.isFinish()) {
            Move move;
            if ((pos.turn_number() + i) % 2 == turn) {
                //思考する
                auto result = searcher.thinkForGenerateLearnData(pos, false);
                move = result.first;

                DWORD dwBytesWritten;
                std::string buffer = fileToString[SquareToFile[move.to()]] + rankToString[SquareToRank[move.to()]];
                if (!WriteFile(pipe_handle, buffer.data(), 10, &dwBytesWritten, NULL)) {
                    fprintf(stderr, "Couldn't write NamedPipe.");
                    std::this_thread::sleep_for(std::chrono::seconds(100));
                    return;
                }
            } else {
                //相手の番
                //入力を待つ
                DWORD dwBytesRead;
                char buffer[10];
                if (!ReadFile(pipe_handle, buffer, 10, &dwBytesRead, NULL)) {
                    fprintf(stderr, "Couldn't read NamedPipe.");
                    std::this_thread::sleep_for(std::chrono::seconds(100));
                    return;
                }
                std::string move_str(buffer);
                move = stringToMove(move_str);
                if (!pos.isLegalMove(move)) {
                    std::cerr << "非合法手: " << move_str << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(100));
                    return;
                }
            }
            pos.doMove(move);
            game.moves.push_back(move);
        }

        //過去の対局と比較して同一でないかを確認する
        bool ok = true;
        for (const auto& g : games) {
            if (g.moves.size() != game.moves.size()) {
                continue;
            }

            bool same = true;
            for (int64_t j = 0; j < (int64_t)g.moves.size(); j++) {
                if (g.moves[j] != game.moves[j]) {
                    same = false;
                    break;
                }
            }
            if (same) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            usi_option.random_turn += 2;
            std::cout << "重複あり random_turn -> " << usi_option.random_turn << std::endl;
            i--;
            continue;
        }

        std::cout << std::setw(4) << i << "局目: ";
        auto result = pos.resultForBlack();
        if (i % 2 == turn) {
            //先手だった
            win_point += result;
            std::cout << result << " 計: " << win_point / (i + 1) << std::endl;
        } else {
            win_point += 1.0 - result;
            std::cout << 1.0 - result << " 計: " << win_point / (i + 1) << std::endl;
        }

        if (turn == 0) {
            game.writeKifuFile("./games/");
        }

        games.push_back(game);
    }

    std::cout << "勝率 : " << win_point / game_num << std::endl;
}