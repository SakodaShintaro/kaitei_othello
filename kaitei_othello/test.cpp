#include"test.hpp"
#include"usi.hpp"
#include"piece.hpp"
#include"square.hpp"
#include"position.hpp"
#include"game.hpp"
#include"network.hpp"
#include"operate_params.hpp"
#include"rootstrap_trainer.hpp"
#include<cassert>
#include<numeric>
#include<set>

void testRandom() {
    //ランダム手
    usi_option.random_turn = 100;

    shared_data.stop_signal = false;
    shared_data.limit_msec = LLONG_MAX;
    Searcher searcher(usi_option.USI_Hash);

    Position pos(*eval_params);

    std::vector<Game> games;
    std::cout << std::fixed;

    for (int64_t i = 0; i < 1000; i++) {
        pos.init();
        Game game;
        while (!pos.isFinish()) {
            //思考する
            auto result = searcher.thinkForGenerateLearnData(pos, false);
            Move move = result.first;
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

        std::cout << std::setw(4) << i << "局目生成終了" << std::endl;
        auto result = pos.resultForBlack();
        games.push_back(game);
    }
}

void testNN() {
    eval_params->readFile();
    Position pos(*eval_params);
    auto searcher = std::make_unique<Searcher>(usi_option.USI_Hash);

    while (!pos.isFinish()) {
        auto moves = pos.generateAllMoves();

        if (moves.front() == NULL_MOVE) {
            pos.doMove(moves.front());
            continue;
        }

        pos.print();

        auto result = searcher->thinkForGenerateLearnData(pos, false);

        std::cout << "探索結果" << std::endl;
        for (Move move : moves) {
            std::cout << move << " " << result.second[move.toLabel()] << std::endl;
        }

        pos.doMove(result.first);
    }
    pos.print();
}

void testKifuOutput() {
    eval_params->readFile();
    Game game;
    Position pos(*eval_params);
    auto searcher = std::make_unique<Searcher>(usi_option.USI_Hash);

    while (!pos.isFinish()) {
        auto result = searcher->thinkForGenerateLearnData(pos, false);
        pos.doMove(result.first);
        game.moves.push_back(result.first);
        game.teachers.push_back(result.second);
    }

    game.writeKifuFile("./");
    printf("finish testKifuOutput()\n");
}

#ifdef USE_CATEGORICAL
void testOneHotDist() {
    std::cout << std::fixed;
    for (double w = 0.0; w <= 1.0; w += 0.01) {
        std::cout << "w = " << w << std::endl;
        auto onehot = onehotDist(w);
        for (int32_t i = 0; i < BIN_SIZE; i++) {
            std::cout << VALUE_WIDTH * i << " ~ " << VALUE_WIDTH * (i + 1) << " : " << onehot[i] << std::endl;
        }
    }
}

void testDistEffect() {
    eval_params->readFile();

    std::ofstream result_fs("dist_effect.txt");

    result_fs << std::fixed;
    std::cout << std::fixed;

    usi_option.random_turn = 30;
    usi_option.thread_num = 1;
    
    auto games = RootstrapTrainer::parallelPlay(*eval_params, *eval_params, 500, 800, false);
    std::set<int64_t> hash_values;

    for (const auto& game : games) {
        Position pos(*eval_params);
        for (auto move : game.moves) {
            if (move == NULL_MOVE) {
                pos.doMove(move);
                continue;
            }

            //move.scoreとPositionのvalueDistの期待値を比べる
            auto value_dist = pos.valueDist();
            
            double value = 0.0;
            for (int32_t i = 0; i < BIN_SIZE; i++) {
                value += VALUE_WIDTH * (i + 0.5) * value_dist[i];
            }

            double sigma = 0.0;
            for (int32_t i = 0; i < BIN_SIZE; i++) {
                sigma += value_dist[i] * pow(VALUE_WIDTH * (i + 0.5) - value, 2);
            }

            if (hash_values.count(pos.hash_value()) == 0) {
                result_fs << move.score << "\t" << value << "\t" << sigma << "\t" << pos.turn_number() << std::endl;
                std::cout << move.score << "\t" << value << "\t" << sigma << "\t" << pos.turn_number() << std::endl;
                hash_values.insert(pos.hash_value());
            }

            pos.doMove(move);
        }
    }
    exit(0);
}

void testTreeDist() {
    eval_params->readFile();

    std::ofstream result_fs("dist_effect.txt");

    result_fs << std::fixed;
    std::cout << std::fixed;

    usi_option.random_turn = 0;
    usi_option.thread_num = 5;
    usi_option.playout_limit = 800;

    auto games = RootstrapTrainer::parallelPlay(*eval_params, *eval_params, 1, 800, false);
    std::set<int64_t> hash_values;

    MCTSearcher searcher(16);

    for (const auto& game : games) {
        Position pos(*eval_params);
        for (auto move : game.moves) {
            if (move == NULL_MOVE) {
                pos.doMove(move);
                continue;
            }

            pos.print();

            auto curr_moves = pos.generateAllMoves();
            for (auto curr_move : curr_moves) {
                //各指し手について探索してみる
                pos.doMove(curr_move);
                auto result = searcher.thinkForGenerateLearnData(pos, false);
                result.first.score = (Score)(1.0 - result.first.score);

                double value = 0.0;
                auto value_dist = pos.valueDist();
                for (int32_t i = 0; i < BIN_SIZE; i++) {
                    value += VALUE_WIDTH * (i + 0.5) * value_dist[i];
                }
                value = 1.0 - value;

                std::cout << "  " << curr_move << ": " << value << ", " << result.first.score << ", " << std::abs(value - result.first.score) << std::endl;

                pos.undo();
            }

            pos.doMove(move);
        }
    }
    exit(0);
}
#endif