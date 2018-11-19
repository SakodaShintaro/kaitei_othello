#pragma once

#include"alhpazero_trainer.hpp"

#include"position.hpp"
#include"searcher.hpp"
#include"eval_params.hpp"
#include"thread.hpp"
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
    //�I�v�V�������t�@�C������ǂݍ���
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
                std::cerr << "optimizer���s��" << std::endl;
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
        } else if (name == "evaluation_random_turn") {
            ifs >> EVALUATION_RANDOM_TURN;
        } else if (name == "policy_loss_coeff") {
            ifs >> POLICY_LOSS_COEFF;
        } else if (name == "value_loss_coeff") {
            ifs >> VALUE_LOSS_COEFF;
        } else if (name == "max_stack_size") {
            ifs >> MAX_STACK_SIZE;
        } else if (name == "max_step_num") {
            ifs >> MAX_STEP_NUM;
#ifdef USE_MCTS
        } else if (name == "playout_limit") {
            ifs >> usi_option.playout_limit;
#endif
        }
    }

    //���̑��I�v�V�������w�K�p�ɐݒ�
    shared_data.limit_msec = LLONG_MAX;
    shared_data.stop_signal = false;
    usi_option.byoyomi_margin = 0LL;
    usi_option.resign_score = MIN_SCORE;

    //�ϐ��̏�����
    update_num_ = 0;
    fail_num_ = 0;
    consecutive_fail_num_ = 0;
    win_average_ = 0.5;

    //�]���֐��ǂݍ���
    eval_params->readFile("tmp.bin");
    eval_params->writeFile("before_learn.bin");

    //Optimizer�ɍ��킹�ĕK�v�Ȃ��̂�����
    if (OPTIMIZER_NAME == "MOMENTUM") {
        pre_update_ = std::make_unique<EvalParams<LearnEvalType>>();
    }

    //������ۑ�����f�B���N�g���̍폜
    std::experimental::filesystem::remove_all("./learn_games");
    std::experimental::filesystem::remove_all("./test_games");

    //������ۑ�����f�B���N�g���̍쐬
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
    start_time_ = std::chrono::steady_clock::now();

    //���O�t�@�C���̐ݒ�
    log_file_.open("alphazero_log.txt");
    print("�o�ߎ���");
    print("�X�e�b�v��");
    print("����");
    print("Policy����");
    print("Value����");
    print("�ő�X�V��");
    print("���a�X�V��");
    print("�ő�p�����[�^");
    print("���a�p�����[�^");
    print("����");
    print("�����z����");
    print("�����z����");
    print("�A�������z����");
    log_file_ << std::endl << std::fixed;
    std::cout << std::endl << std::fixed;

    //���ȑ΋ǃX���b�h�̍쐬
    std::vector<std::thread> slave_threads(THREAD_NUM - 1);
    for (uint32_t i = 0; i < THREAD_NUM - 1; i++) {
        slave_threads[i] = std::thread(&AlphaZeroTrainer::learnSlave, this);
    }

    //�������\�����܂�܂ő҂�
    //busy wait!
    //while (position_stack_.size() <= 0) {
    //    continue;
    //}

    //�w�K����
    std::random_device seed;
    std::default_random_engine engine(seed());

    Position pos(*eval_params);
    for (int32_t step_num = 1; step_num <= MAX_STEP_NUM; step_num++) {
        //�~�j�o�b�`�����z�𒙂߂�
        auto grad = std::make_unique<EvalParams<LearnEvalType>>();
        std::array<double, 2> loss{ 0.0, 0.0 };
        for (int32_t j = 0; j < BATCH_SIZE; j++) {
            if (position_stack_.size() <= BATCH_SIZE * 20) {
                j--;
                continue;
            }

            //�����_���ɑI��
            MUTEX.lock();
            int32_t random = engine() % position_stack_.size();
            auto data = position_stack_[random];
            MUTEX.unlock();

            //�ǖʂ𕜌�
            pos.loadData(data.first);
            
            //���z���v�Z
            loss += addGrad(*grad, pos, data.second);
        }
        loss /= BATCH_SIZE;
        grad->forEach([this](CalcType& g) {
            g /= BATCH_SIZE;
        });

        MUTEX.lock();
        //�w�K
        updateParams(*eval_params, *grad);

        //�p�����[�^�X�V
        updateParams(*eval_params, *grad);

        //�����o��
        eval_params->writeFile("tmp" + std::to_string(step_num) + ".bin");

        //�w�K���̕\��
        timestamp();
        print(step_num);
        print(POLICY_LOSS_COEFF * loss[0] + VALUE_LOSS_COEFF * loss[1]);
        print(loss[0]);
        print(loss[1]);
        print(LEARN_RATE * grad->maxAbs());
        print(LEARN_RATE * grad->sumAbs());
        print(eval_params->maxAbs());
        print(eval_params->sumAbs());

        //��������x����
        //LEARN_RATE *= LEARN_RATE_DECAY;

        //�]��
        if (step_num % EVALUATION_INTERVAL == 0) {
            evaluate();
        }

        if (step_num % 100 == 0) {
            eval_params->writeFile("tmp" + std::to_string(0) + "_" + std::to_string(step_num) + ".bin");
        }

        std::cout << std::endl;
        log_file_ << std::endl;

        MUTEX.unlock();
    }

    shared_data.stop_signal = true;
    for (uint32_t i = 0; i < THREAD_NUM - 1; i++) {
        slave_threads[i].join();
        printf("%2d�X���b�h��join\n", i);
    }

    log_file_.close();
    std::cout << "finish learnAsync()" << std::endl;
}

void AlphaZeroTrainer::learnSlave() {
    //��~�M��������܂Ń��[�v
    while (!shared_data.stop_signal) {
        //�����𐶐�
#ifdef USE_MCTS
        auto games = play(1, (int32_t)usi_option.playout_limit, true);
#else
        auto games = play(1, SEARCH_DEPTH);
#endif

        MUTEX.lock();
        //���������������w�K�p�f�[�^�ɉ��H����stack�֋l�ߍ���
        for (const auto& game : games) {
            Position pos(*eval_params);
            for (int32_t i = 0; !pos.isFinish(); i++) {
                if (game.moves[i] == NULL_MOVE) {
                    //�w�K�ɂ͎g��Ȃ�.�ǖʂ�i�߂Ď���
                    pos.doMove(game.moves[i]);
                    continue;
                }

                //���̋ǖʂɂ��ċǖʂ��Č��ł���f�[�^�Ƌ��t�f�[�^��g�݂ɂ���stack�ɑ���
                position_stack_.push_back({ pos.data(), game.teachers[i] });

                //���̋ǖʂ�
                pos.doMove(game.moves[i]);
            }
        }
        if (position_stack_.size() >= MAX_STACK_SIZE) {
            auto diff = position_stack_.size() - MAX_STACK_SIZE;
            position_stack_.erase(position_stack_.begin(), position_stack_.begin() + diff);
        }
        MUTEX.unlock();
    }
}

std::vector<Game> AlphaZeroTrainer::play(int32_t game_num, int32_t search_limit, bool add_noise) {
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
            //i�������̂Ƃ�pos_c�����
            auto move_and_teacher = searcher->thinkForGenerateLearnData(pos, search_limit, add_noise);
            Move best_move = move_and_teacher.first;
            TeacherType teacher = move_and_teacher.second;

            if (best_move != NULL_MOVE && !pos.isLegalMove(best_move)) {
                pos.printForDebug();
                best_move.printWithScore();
                assert(false);
            }
            pos.doMove(best_move);
            game.moves.push_back(best_move);
            game.teachers.push_back(teacher);
        }

        //�΋ǌ��ʂ̐ݒ�
        game.result = pos.resultForBlack();
    }
    return games;
}

std::vector<Game> AlphaZeroTrainer::parallelPlay(const EvalParams<DefaultEvalType>& curr, const EvalParams<DefaultEvalType>& target, int32_t game_num, int32_t search_limit, bool add_noise) {
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
                    //i�������̂Ƃ�pos_c�����
                    auto move_and_teacher = ((pos_c.turn_number() % 2) == (curr_index % 2) ?
                        searcher->thinkForGenerateLearnData(pos_c, search_limit, add_noise) :
                        searcher->thinkForGenerateLearnData(pos_t, search_limit, add_noise));
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

                //�΋ǌ��ʂ̐ݒ�
                game.result = pos_c.resultForBlack();
            }
        });
    }
    for (int32_t i = 0; i < (int32_t)usi_option.thread_num; i++) {
        threads[i].join();
    }
    return games;
}

void AlphaZeroTrainer::evaluate() {
    //�΋ǂ���p�����[�^������
    auto opponent_parameters_ = std::make_unique<EvalParams<DefaultEvalType>>();
    opponent_parameters_->readFile();

    //random_turn�͏����߂ɂ���
    auto copy = usi_option.random_turn;
    usi_option.random_turn = (uint32_t)EVALUATION_RANDOM_TURN;
#ifdef USE_MCTS
    auto test_games = parallelPlay(*eval_params, *opponent_parameters_, EVALUATION_GAME_NUM, (int32_t)usi_option.playout_limit, false);
#else
    auto test_games = parallelPlay(*eval_params, *opponent_parameters_, EVALUATION_GAME_NUM, SEARCH_DEPTH);
#endif
    usi_option.random_turn = copy;

    //�������o��
    for (int32_t i = 0; i < std::min(4, (int32_t)test_games.size()); i++) {
        test_games[i].writeKifuFile("./test_games/");
    }

    double win_rate = 0.0;
    for (int32_t i = 0; i < test_games.size(); i++) {
        win_rate += (i % 2 == 0 ? test_games[i].result : 1.0 - test_games[i].result);
    }

    //�d���̊m�F�����Ă݂�
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
    printf("%d\t%d\t", same_num, (same_num == 0 ? EVALUATION_RANDOM_TURN : ++EVALUATION_RANDOM_TURN));
    win_rate /= test_games.size();

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

std::array<double, 2> AlphaZeroTrainer::learnGames(const std::vector<Game>& games, EvalParams<LearnEvalType>& grad) {
    std::array<double, 2> loss = { 0.0, 0.0 };
    grad.clear();

    //���������������ꍇ������̂ł����BATCH_SIZE�Ɉ�v����Ƃ͌���Ȃ�
    uint64_t learn_position_num = 0;

    //������o���Ă݂�
    games.front().writeKifuFile("./learn_games/");

    for (const Game& game : games) {
        //�w�K
        if (LEARN_MODE == ELMO_LEARN) {
            learnOneGame(game, grad, loss, learn_position_num);
        } else if (LEARN_MODE == N_STEP_SARSA) {
            learnOneGameReverse(game, grad, loss, learn_position_num);
        } else { //�����ɂ͗��Ȃ��͂�
            assert(false);
        }
    }

    assert(learn_position_num != 0);

    //loss, grad��learn_position_num�Ŋ���(�ǖʂɂ��ĕ��ς����)
    loss /= learn_position_num;

    grad.forEach([learn_position_num](CalcType& g) {
        g /= learn_position_num;
    });

    return loss;
}

void AlphaZeroTrainer::learnOneGame(const Game& game, EvalParams<LearnEvalType>& grad, std::array<double, 2>& loss, uint64_t& learn_position_num) {
    Position pos(*eval_params);
#ifndef USE_NN //����T����@�̈Ⴂ����ˁH
    auto searcher = std::make_unique<Searcher>(Searcher::SLAVE);
#endif
    for (int32_t i = 0; i < game.moves.size(); i++) {
        Move m = game.moves[i];
        if (m == NULL_MOVE) {
            pos.doMove(m);
            continue;
        }

        //�w�K
        learn_position_num++;
        TeacherType teacher = game.teachers[i];
        //�΋ǌ��ʂ�p����value�����H����
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
        //�����E���z�̌v�Z
        loss += addGrad(grad, pos, teacher);

        //���l�����ɂ��덷�t�`�d�̌���
        //verifyAddGrad(pos, teacher);

        if (!pos.isLegalMove(m)) {
            pos.printForDebug();
            m.printWithScore();
        }

        pos.doMove(m);
    }
}

void AlphaZeroTrainer::learnOneGameReverse(const Game& game, EvalParams<LearnEvalType>& grad, std::array<double, 2>& loss, uint64_t& learn_position_num) {
    auto searcher = std::make_unique<Searcher>(Searcher::SLAVE);
    Position pos(*eval_params);

    //�܂��͍ŏI�ǖʂ܂œ�����
    for (auto move : game.moves) {
        pos.doMove(move);
    }

    assert(Game::RESULT_WHITE_WIN <= game.result && game.result <= Game::RESULT_BLACK_WIN);

    //��肩�猩������,���z.�w���ړ����ςœ������Ă���.�ŏ��͌��ʂɂ���ď�����(0 or 0.5 or 1)
    double win_rate_for_black = game.result;
#ifdef USE_CATEGORICAL
    auto dist_for_black = onehotDist(win_rate_for_black);
#endif

    for (int32_t i = (int32_t)game.moves.size() - 1; i >= 0; i--) {
        if (game.moves[i].score == MIN_SCORE) { //�����_�����[�u�Ƃ������ƂȂ̂Ŋw�K�͂��Ȃ�
            //�����_�����[�u��1�ǂ̍ŏ��̕��ɍs���Ă���̂ł����w�K�I��
            break;
        }

        //���̎w���肪�Ή�����̂�1��߂����ǖ�
        pos.undo();

        if (isMatedScore(game.moves[i].score) || game.moves[i] == NULL_MOVE) { //�l�݂̒l��������w�K���΂�
            continue;
        }

        //��肩�猩�������ɂ��Ďw���ړ����ς����,���t�f�[�^�ɃZ�b�g����
        //���t�f�[�^���R�s�[���� game��const�Ŏ󂯎���Ă��܂��Ă���̂�
        TeacherType teacher = game.teachers[i];

#ifdef USE_CATEGORICAL
        //teacher���番�z�𓾂�
        std::array<CalcType, BIN_SIZE> curr_dist;
        for (int32_t j = 0; j < BIN_SIZE; j++) {
            curr_dist[j] = teacher[POLICY_DIM + j];
        }

        //��Ԃ��l�����Đ�肩�猩�����z�ɂ���
        if (pos.color() == WHITE) {
            std::reverse(curr_dist.begin(), curr_dist.end());
        }

        //��������
        for (int32_t j = 0; j < BIN_SIZE; j++) {
            dist_for_black[j] = (CalcType)(DECAY_RATE * dist_for_black[j] + (1.0 - DECAY_RATE) * curr_dist[j]);
        }

        //teacher�ɃR�s�[����
        for (int32_t j = 0; j < BIN_SIZE; j++) {
            teacher[POLICY_DIM + j] = dist_for_black[j];
        }

        //��Ԃɍ��킹�Ĕ��]����
        if (pos.color() == WHITE) {
            std::reverse(teacher.begin() + POLICY_DIM, teacher.end());
        }
#else
        //��肩�猩���l�𓾂�
        double curr_win_rate = (pos.color() == BLACK ? teacher[POLICY_DIM] : 1.0 - teacher[POLICY_DIM]);

        //��������
        win_rate_for_black = DECAY_RATE * win_rate_for_black + (1.0 - DECAY_RATE) * curr_win_rate;

        //teacher�ɃR�s�[����
        teacher[POLICY_DIM] = (CalcType)(pos.color() == BLACK ? win_rate_for_black : 1.0 - win_rate_for_black);
#endif
        //�����E���z�̌v�Z
        loss += addGrad(grad, pos, teacher);

        //���l�����ɂ��덷�t�`�d�̌���
        //verifyAddGrad(pos, teacher);

        //�w�K�ǖʐ��𑝂₷
        learn_position_num++;
    }
}

void AlphaZeroTrainer::testLearn() {
    std::cout << "start testLearn()" << std::endl;

    //���Ԃ�ݒ�
    start_time_ = std::chrono::steady_clock::now();

    //���ȑ΋ǂɂ���������:����
#ifdef USE_MCTS
    std::vector<Game> games = parallelPlay(*eval_params, *eval_params, BATCH_SIZE, (int32_t)usi_option.playout_limit, true);
#else
    std::vector<Game> games = parallelPlay(*eval_params, *eval_params, BATCH_SIZE, SEARCH_DEPTH);
#endif

    std::cout << std::fixed;

    //��������w�K�̃��C��
    eval_params->readFile();

    std::ofstream ofs("test_learn_log.txt");
    ofs << "step\tP = " << POLICY_LOSS_COEFF << ", V = " << VALUE_LOSS_COEFF << ", LEARN_RATE = " << LEARN_RATE << std::endl;

    for (int64_t i = 0; i < 1000; i++) {
        //�����E���z�E����萔�E���萔�ɂ��������������v�Z
        auto grad = std::make_unique<EvalParams<LearnEvalType>>();
        std::array<double, 2> loss = learnGames(games, *grad);

        //�p�����[�^�X�V
        updateParams(*eval_params, *grad);
        std::cout << i << "\tloss[0] = " << loss[0] << ",\tloss[1] = " << loss[1] << std::endl;
        ofs << i << "\t" << loss[0] << "\t" << loss[1] << std::endl;
    }

    for (auto game : games) {
        Position pos(*eval_params);
        std::cout << "game.result = " << game.result << std::endl;

        for (int32_t i = 0; i < game.moves.size(); i++) {
            auto move = game.moves[i];
            if (move != NULL_MOVE) {
                auto policy = pos.maskedPolicy();
                std::cout << "policy[" << std::setw(4) << move << "] = " << policy[move.toLabel()]
                    << ", value = " << pos.valueForTurn()
                    << ", teacher = " << game.teachers[i][POLICY_DIM] << std::endl;
            }
            pos.doMove(move);
        }
    }

    std::cout << "finish testLearn()" << std::endl;
}