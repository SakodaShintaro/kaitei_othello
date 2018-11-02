#include"test.hpp"
#include"usi.hpp"
#include"piece.hpp"
#include"square.hpp"
#include"position.hpp"
#include"game.hpp"
#include"network.hpp"
#include<cassert>
#include<numeric>

void testMakeRandomPosition() {
    std::unique_ptr<Searcher> s(new Searcher(Searcher::SLAVE));

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

    usi_option.draw_turn = 256;

    //同一局面の登場回数
    std::map<uint64_t, uint64_t> appear_num;

    //評価値
    std::vector<double> scores(try_num);

    for (uint64_t i = 0; i < try_num; i++) {
        Position p(*eval_params);

        int move_count = 0;
        for (uint64_t j = 0; j < random_turn; j++) {
            Move random_move = s->softmaxChoice(p, temperature);
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

    while (true) {
        auto moves = pos.generateAllMoves();
        if (moves.size() == 0) {
            break;
        }
        pos.print();

        auto feature = pos.makeFeatures();
        for (int32_t r = Rank1; r <= Rank9; r++) {
            for (int32_t f = File9; f >= File1; f--) {
                auto sq = FRToSquare[f][r];
                printf("%5d ", (int)feature[SquareToNum[sq]]);
            }
            printf("\n");
        }

        auto result = pos.policy();
        printf("value_score = %f\n", pos.valueScore());

        Network::scoreByPolicy(moves, result, 10000);
        sort(moves.begin(), moves.end(), std::greater<Move>());
        for (auto move : moves) {
            move.printWithScore();
        }

        pos.doMove(moves.front());
    }
}

void testKifuOutput() {
    Game game;
    eval_params->readFile();
    Position pos_c(*eval_params), pos_t(*eval_params);
    auto searcher = std::make_unique<Searcher>(Searcher::SLAVE);
    usi_option.draw_turn = 512;

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