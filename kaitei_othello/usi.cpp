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
#include"test.hpp"
#include"network.hpp"
#include"MCTSearcher.hpp"
#include"operate_params.hpp"
#include<iostream>
#include<string>
#include<thread>

USIOption usi_option;

void USI::loop() {
    std::string input;
    while (true) {
        std::cin >> input;
        if (input == "usi") {
            usi();
        } else if (input == "isready") {
            isready();
        } else if (input == "setoption") {
            setoption();
        } else if (input == "usinewgame") {
            usinewgame();
        } else if (input == "position") {
            position();
        } else if (input == "go") {
            go();
        } else if (input == "stop") {
            stop();
        } else if (input == "ponderhit") {
            ponderhit();
        } else if (input == "quit") {
            quit();
        } else if (input == "gameover") {
            gameover();
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
        } else if (input == "printEvalParams") {
            eval_params->readFile();
            eval_params->printHistgram();
        } else if (input == "testMakeRandomPosition") {
            testMakeRandomPosition();
        } else if (input == "testKifuOutput") {
            testKifuOutput();
        } else if (input == "testNN") {
            testNN();
        } else if (input == "testLearn") {
            RootstrapTrainer trainer("rootstrap_settings.txt");
            trainer.testLearn();
        } else if (input == "vsHuman") {
            vsHuman();
        } else {
            std::cout << "Illegal input" << std::endl;
        }
    }
}

void USI::usi() {
    printf("id name kaitei_othello_nn\n");
    printf("id author Sakoda Shintaro\n");
	printf("option name byoyomi_margin type spin default 0 min 0 max 1000\n");
    usi_option.byoyomi_margin = 0;
	printf("option name random_turn type spin default 0 min 0 max 100\n");
    usi_option.random_turn = 0;
    printf("option name thread_num type spin default 1 min 1 max %d\n", std::max(std::thread::hardware_concurrency(), 1U));
    usi_option.thread_num = 1;
    printf("option name draw_score type spin default -1 min -30000 max 30000\n");
    usi_option.draw_score = -1;
    printf("option name draw_turn type spin default 256 min 0 max 4096\n");
    usi_option.draw_turn = 256;
    printf("option name temperature type spin default 10 min 1 max 100000\n");
    usi_option.temperature = 10.0;
    printf("option name resign_score type spin default %d min %d max %d\n", (int32_t)MIN_SCORE, (int32_t)MIN_SCORE, (int32_t)MAX_SCORE);
    usi_option.resign_score = MIN_SCORE;

#ifdef USE_MCTS
    uint64_t d = (uint64_t)1e10;
    printf("option name playout_limit type spin default %llu min 1 max %llu\n", d, d);
    usi_option.playout_limit = d;
#endif

    usi_option.USI_Hash = 256;
	printf("usiok\n");
}

void USI::isready() {
    shared_data.hash_table.setSize(usi_option.USI_Hash);
    eval_params->readFile();
    printf("readyok\n");
}

void USI::setoption() {
    std::string input;
    while (true) {
        std::cin >> input;
        if (input == "name") {
            std::cin >> input;
            //ここで処理
            if (input == "byoyomi_margin") {
                std::cin >> input; //input == "value"となるなず
                std::cin >> usi_option.byoyomi_margin;
                return;
            } else if (input == "random_turn") {
                std::cin >> input; //input == "value"となるなず
                std::cin >> usi_option.random_turn;
                return;
            } else if (input == "USI_Hash") {
                std::cin >> input; //input == "value"となるはず
                std::cin >> usi_option.USI_Hash;
                return;
            } else if (input == "USI_Ponder") {
                std::cin >> input; //input == "value"となるなず
                std::cin >> input; //特になにもしていない
                return;
            } else if (input == "thread_num") {
                std::cin >> input; //input == "value"となるはず
                std::cin >> usi_option.thread_num;
                threads.clear();
                threads.init();
                return;
            } else if (input == "draw_score") {
                std::cin >> input; //input == "value"となるはず
                std::cin >> usi_option.draw_score;
                return;
            } else if (input == "draw_turn") {
                std::cin >> input; //input == "value"となるはず
                std::cin >> usi_option.draw_turn;
                return;
            } else if (input == "temperature") {
                std::cin >> input; //input == "value"となるはず
                std::cin >> usi_option.temperature;
                return;
            } else if (input == "resign_score") {
                std::cin >> input;
                std::cin >> usi_option.resign_score;
                return;
#ifdef USE_MCTS
            } else if (input == "playout_limit") {
                std::cin >> input;
                std::cin >> usi_option.playout_limit;
                return;
#endif
            }
        }
    }
}

void USI::usinewgame() {
    //置換表のリセット
    shared_data.hash_table.clear();
    shared_data.hash_table.setSize(usi_option.USI_Hash);
}

void USI::position() {
    //rootを初期化
    shared_data.root.init();

    std::string input, sfen;
    std::cin >> input;
    if (input == "startpos") {
        sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
    } else {
        for (int i = 0; i < 4; i++) {
            std::cin >> input;
            sfen += input;
            sfen += " ";
        }
    }

    std::cin >> input;  //input == "moves" or "go"となる
    if (input != "go") {
        while (std::cin >> input) {
            if (input == "go") {
                break;
            }
            //inputをMoveに直して局面を動かす
            Move move = stringToMove(input);
            shared_data.root.doMove(move);
        }
    }

    go();
}

void USI::go() {
    shared_data.stop_signal = false;
    std::string input;
	int64_t btime, wtime, byoyomi = 0, binc = 0, winc = 0;
#ifdef USE_MCTS
    MCTSearcher mctsearcher(usi_option.USI_Hash);
#endif
    std::cin >> input;
    if (input == "ponder") {
        //ponderの処理
    } else if (input == "btime") {
        std::cin >> input;
        btime = atoi(input.c_str());
        std::cin >> input; //input == "wtime" となるはず
        std::cin >> input;
        wtime = atoi(input.c_str());
        std::cin >> input; //input == "byoyomi" or "binc"となるはず
        if (input == "byoyomi") {
            std::cin >> input;
            byoyomi = atoi(input.c_str());
        } else {
            std::cin >> input;
            binc = atoi(input.c_str());
            std::cin >> input; //input == "winc" となるはず
            std::cin >> input;
            winc = atoi(input.c_str());
        }
        //ここまで持ち時間の設定
        //ここから思考部分
        if (byoyomi != 0) {
            shared_data.limit_msec = byoyomi;
        } else {
            shared_data.limit_msec = binc;
        }
#ifdef USE_MCTS
        mctsearcher.think();
#else
        threads[0]->startSearch();
#endif
        return;
    } else if (input == "infinite") {
        //stop来るまで思考し続ける
        //思考時間をほぼ無限に
        shared_data.limit_msec = LLONG_MAX;
        
        //randomturnをなくす
        usi_option.random_turn = 0;

#ifdef USE_MCTS
        mctsearcher.think();
#else
        threads[0]->startSearch();
#endif
    } else if (input == "mate") {
        //詰み探索
        std::cin >> input;
        if (input == "infinite") {
            //stop来るまで
        } else {
            //思考時間が指定された場合
            //どう実装すればいいんだろう
        }
    }
}

void USI::stop() {
    shared_data.stop_signal = true;
    for (auto& t : threads) {
        t->waitForFinishSearch();
    }
}

void USI::ponderhit() {
    //まだ未実装
}

void USI::quit() {
    stop();
    exit(0);
}

void USI::gameover() {
    std::string input;
    std::cin >> input;
    if (input == "win") {
        //勝ち
    } else if (input == "lose") {
        //負け
    } else if (input == "draw") {
        //引き分け
		return;
	}
}

void USI::vsHuman() {
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
            std::cout << "指し手を入力(将棋形式で筋と段をスペース区切り): ";
            int32_t file, rank;
            std::cin >> file >> rank;
            pos.doMove(Move(FRToSquare[file][rank]));
        } else {
            auto result = mctsearcher.thinkForGenerateLearnData(pos, (int32_t)usi_option.playout_limit);
            pos.doMove(result.first);
        }
    }
}