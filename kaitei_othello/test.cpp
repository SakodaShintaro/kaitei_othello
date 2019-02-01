#include"test.hpp"
#include"usi.hpp"
#include"piece.hpp"
#include"square.hpp"
#include"position.hpp"
#include"game.hpp"
#include"network.hpp"
#include"operate_params.hpp"
#include"alphazero_trainer.hpp"
#include<cassert>
#include<numeric>
#include<iomanip>
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
            auto result = searcher.think(pos, false);
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

        auto result = searcher->think(pos, false);

        if (pos.turn_number() % 10 != 0) {
            pos.doMove(result.first);
            continue;
        }

        pos.print();

#ifdef USE_CATEGORICAL
        auto dist = pos.valueDist();
        bool print_dist;
        std::cout << "表示? ";
        std::cin >> print_dist;
        if (print_dist) {
            for (int32_t i = 0; i < BIN_SIZE; i++) {
                std::cout << dist[i] << std::endl;
            }
        }
#endif

        std::cout << "探索結果" << std::endl;
        for (Move move : moves) {
            std::cout << move << " " << result.second[move.toLabel()] << std::endl;
        }

        pos.doMove(result.first);
    }
    pos.print();
    std::cout << pos.scoreForBlack() << std::endl;
}

void testKifuOutput() {
    eval_params->readFile();
    Game game;
    Position pos(*eval_params);
    auto searcher = std::make_unique<Searcher>(usi_option.USI_Hash);

    while (!pos.isFinish()) {
        auto result = searcher->think(pos, false);
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
#ifdef USE_MCTS
    usi_option.playout_limit = 800;
#else
    usi_option.depth_limit = 8;
    usi_option.node_limit = 800;
#endif
    
    auto games = AlphaZeroTrainer::parallelPlay(*eval_params, *eval_params, 100, false);
    std::set<int64_t> hash_values;

    result_fs << "探索した値";
    constexpr int32_t MAX = 25;
    for (int32_t i = 0; i <= MAX; i++) {
        result_fs << "\t幅" << 2 * i + 1 << "確率";
    }
    result_fs << std::endl;

    auto isOK = [&](int32_t i) {
        return 0 <= i && i < BIN_SIZE;
    };

    for (const auto& game : games) {
        Position pos(*eval_params);
        for (auto move : game.moves) {
            if (move == NULL_MOVE || hash_values.count(pos.hash_value())) {
                pos.doMove(move);
                continue;
            }

            result_fs << move.score;

            //move.score
            auto value_dist = pos.valueDist();
            
            int32_t index = valueToIndex(move.score);


            for (int32_t i = 0; i <= MAX; i++) {
                //indexから左右i個までを考える
                double p = value_dist[index];

                int32_t num = 2 * i;

                //左へ進む
                for (int32_t j = 1; j <= i && isOK(index - j); j++, num--) {
                    assert(isOK(index - j));
                    p += value_dist[index - j];
                    //std::cout << "value_dist[" << index - j << "] = " << value_dist[index - j] << ", p = " << p << std::endl;
                }

                //右へ進む
                for (int32_t j = 1; num > 0 && isOK(index + j); j++, num--) {
                    assert(isOK(index + j));
                    p += value_dist[index + j];
                    //std::cout << "value_dist[" << index + j << "] = " << value_dist[index + j] << ", p = " << p << std::endl;
                }

                //左へ進む
                for (int32_t j = i + 1; num > 0; j++, num--) {
                    if (!isOK(index - j)) {
                        std::cout << p << std::endl;
                        std::cout << "index = " << index << ", i = " << i << std::endl;
                        assert(false);
                    }
                    p += value_dist[index - j];
                    //std::cout << "value_dist[" << index - j << "] = " << value_dist[index - j] << ", p = " << p << std::endl;
                }

                if (!(0.0 <= p && p <= 1.1)) {
                    std::cout << p << std::endl;
                    std::cout << "index = " << index << ", i = " << i << std::endl;
                    assert(false);
                }
                result_fs << "\t" << p;
            }
            result_fs << std::endl;

            hash_values.insert(pos.hash_value());

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

    auto games = AlphaZeroTrainer::parallelPlay(*eval_params, *eval_params, 1, false);
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
                auto result = searcher.think(pos, false);
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