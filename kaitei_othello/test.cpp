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

void testMakeRandomPosition() {
    auto s = std::make_unique<Searcher>(usi_option.USI_Hash);

    uint64_t try_num, random_turn, SEARCH_DEPTH;
    double temperature;
    std::cout << "何回試行するか:";
    std::cin >> try_num;
    std::cout << "何手ランダムの局面か:";
    std::cin >> random_turn;
    std::cout << "ランダムムーブ後の局面を調べる深さ:";
    std::cin >> SEARCH_DEPTH;
    std::cout << "温度:";
    std::cin >> temperature;
    std::cout << std::endl;

    //同一局面の登場回数
    std::map<uint64_t, uint64_t> appear_num;

    //評価値
    std::vector<double> scores(try_num);

    for (uint64_t i = 0; i < try_num; i++) {
        Position p(*eval_params);

        int move_count = 0;
        for (uint64_t j = 0; j < random_turn; j++) {
            Move random_move = s->thinkForGenerateLearnData(p, true).first;
            if (random_move == NULL_MOVE) {
                break;
            }
            p.doMove(random_move);
            move_count++;
        }

        //手数分ランダムに動かした
        //p.print();
        
        if (move_count != random_turn) {
            //途中で詰んだ場合もう一度やり直す
            for (uint64_t j = 0; j < move_count; j++) {
                p.undo();
            }
            i--;
            continue;
        }

        //探索してランダムムーブ後の局面の評価値を得る
        auto result = s->thinkForGenerateLearnData(p, Depth(SEARCH_DEPTH * (int)PLY));
        auto move = result.first;
        if (isMatedScore(move.score)) {
            //探索して詰みが見えた場合ももう一度
            for (uint64_t j = 0; j < move_count; j++) {
                p.undo();
            }
            i--;
            continue;
        }
        scores[i] = static_cast<double>(move.score);
        //std::cout << "今回のスコア : " << scores[i] << std::endl;
        appear_num[p.hash_value()]++;
    }

    //統計情報を得る
    double sum = std::accumulate(scores.begin(), scores.end(), 0.0);
    double average = sum / try_num;
    double variance = 0;
    for (double s : scores) {
        variance += std::pow(s - average, 2);
    }
    variance /= try_num;

    printf("average = %f, variance = %f\n", average, variance);

    //局面の重複を確認
    for (auto p : appear_num) {
        if (p.second >= 2) {
            std::cout << p.first << "の局面の登場回数:" << p.second << std::endl;
        }
    }

    printf("終了\n");
}

void testNN() {
    eval_params->readFile();
    Position pos(*eval_params);

    while (!pos.isFinish()) {
        auto moves = pos.generateAllMoves();

        if (moves.front() == NULL_MOVE) {
            pos.doMove(moves.front());
            continue;
        }

        pos.print();

        auto policy = pos.maskedPolicy();

        for (auto& move : moves) {
            move.score = policy[move.toLabel()];
        }
        sort(moves.begin(), moves.end(), std::greater<Move>());
        for (const auto& move : moves) {
            move.printWithScore();
        }

        pos.doMove(moves.front());
    }
    pos.print();
}

void testKifuOutput() {
    Game game;
    eval_params->readFile();
    Position pos_c(*eval_params), pos_t(*eval_params);
    auto searcher = std::make_unique<Searcher>(usi_option.USI_Hash);

    while (true) {
        //iが偶数のときpos_cが先手
        auto move_and_teacher = ((pos_c.turn_number() % 2) == 0 ?
            searcher->thinkForGenerateLearnData(pos_c, 3) :
            searcher->thinkForGenerateLearnData(pos_t, 3));
        Move best_move = move_and_teacher.first;
        TeacherType teacher = move_and_teacher.second;

        if (best_move == NULL_MOVE) { //NULL_MOVEは投了を示す
            game.result = (pos_c.color() == BLACK ? Game::RESULT_WHITE_WIN : Game::RESULT_BLACK_WIN);
            break;
        }

        if (!pos_c.isLegalMove(best_move)) {
            pos_c.printForDebug();
            best_move.printWithScore();
            assert(false);
        }
        pos_c.doMove(best_move);
        pos_t.doMove(best_move);
        game.moves.push_back(best_move);
        game.teachers.push_back(teacher);
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
                result.first.score = 1.0 - result.first.score;

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