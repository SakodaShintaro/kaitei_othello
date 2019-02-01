#pragma once

#include"alphazero_trainer.hpp"
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

//�ۑ�����f�B���N�g���̖��O
static const std::string LEARN_GAMES_DIR = "./learn_games/";
static const std::string EVAL_GAMES_DIR = "./test_games/";

AlphaZeroTrainer::AlphaZeroTrainer(std::string settings_file_path) {
    //�I�v�V�������t�@�C������ǂݍ���
    std::ifstream ifs(settings_file_path);
    if (!ifs) {
        std::cerr << "fail to open setting_file(" << settings_file_path << ")" << std::endl;
        assert(false);
    }

    replay_buffer_ = std::make_unique<ReplayBuffer>();

    std::string name;
    while (ifs >> name) {
        if (name == "batch_size") {
            ifs >> BATCH_SIZE;
        } else if (name == "optimizer") {
            ifs >> OPTIMIZER_NAME;
            if (!isLegalOptimizer()) {
                std::cerr << "optimizer���s��" << std::endl;
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
        } else if (name == "random_move_num") {
            ifs >> usi_option.random_turn;
        } else if (name == "lambda") {
            ifs >> replay_buffer_->LAMBDA;
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
            ifs >> replay_buffer_->MAX_STACK_SIZE;
        } else if (name == "max_step_num") {
            ifs >> MAX_STEP_NUM;
        } else if (name == "wait_limit_size") {
            ifs >> replay_buffer_->WAIT_LIMIT_SIZE;
        } else if (name == "wait_coeff") {
            ifs >> WAIT_COEFF;
        } else if (name == "learn_num") {
            ifs >> LEARN_NUM;
        } else if (name == "print_interval") {
            ifs >> PRINT_INTERVAL;
#ifdef USE_MCTS
        } else if (name == "playout_limit") {
            ifs >> usi_option.playout_limit;
#endif
        }
    }

    //���̑��I�v�V�������w�K�p�ɐݒ�
    shared_data.limit_msec = LLONG_MAX;
    shared_data.stop_signal = false;

    //Optimizer�ɍ��킹�ĕK�v�Ȃ��̂�����
    if (OPTIMIZER_NAME == "MOMENTUM") {
        pre_update_ = std::make_unique<EvalParams<LearnEvalType>>();
    }

    //������ۑ�����f�B���N�g���̍폜
    std::experimental::filesystem::remove_all(LEARN_GAMES_DIR);
    std::experimental::filesystem::remove_all(EVAL_GAMES_DIR);

    //������ۑ�����f�B���N�g���̍쐬
#ifdef _MSC_VER
    _mkdir(LEARN_GAMES_DIR.c_str());
    _mkdir(EVAL_GAMES_DIR.c_str());
#elif __GNUC__
    mkdir(LEARN_GAMES_DIR.c_str(), ACCESSPERMS);
    mkdir(EVAL_GAMES_DIR.c_str(), ACCESSPERMS);
#endif
}

void AlphaZeroTrainer::learn() {
    std::cout << "start alphaZero()" << std::endl;

    //���ȑ΋ǃX���b�h�̍쐬
    std::vector<std::thread> slave_threads(THREAD_NUM - 1);
    for (uint32_t i = 0; i < THREAD_NUM - 1; i++) {
        slave_threads[i] = std::thread(&AlphaZeroTrainer::learnSlave, this);
    }

    //�����̏���
    std::random_device seed;
    std::default_random_engine engine(seed());

    //�ǖʂ��C���X�^���X�͈�p�ӂ��ēs�x�ǖʂ��\��
    Position pos(*eval_params);

    //�����������ĉ󂵂Ă��܂��̂Ŋw�K���̏����l��ێ����Ă���
    auto start_learning_rate = LEARN_RATE;

    //model.bin��first_target.bin�փR�s�[
    eval_params->readFile();
    eval_params->writeFile("first_target.bin");

    //�w�K
    for (int32_t i = 1; i <= LEARN_NUM; i++) {
        //���Ԃ�������
        start_time_ = std::chrono::steady_clock::now();

        //first_target��model.bin�փR�s�[
        eval_params->readFile("first_target.bin");
        eval_params->writeFile();

        //�p�����[�^�̏�����
        eval_params->initRandom();
        eval_params->writeFile("before_learn" + std::to_string(i) + ".bin");

        //�ϐ��̏�����
        update_num_ = 0;

        //�w�K���̏�����
        LEARN_RATE = start_learning_rate;

        //���O�t�@�C���̐ݒ�
        log_file_.open("alphazero_log" + std::to_string(i) + ".txt");
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
        print("�X�V��");
        print("�d����");
        print("����̃����_���萔");
        log_file_ << std::endl << std::fixed;
        std::cout << std::endl << std::fixed;

        //0��ڂ����Ă݂�
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
        print("\n", false);

        replay_buffer_->clear();

        for (int32_t step_num = 1; step_num <= MAX_STEP_NUM; step_num++) {
            //���̃X�e�b�v�ɂ����������Ԃ�������
            auto step_start = std::chrono::steady_clock::now();

            //�~�j�o�b�`�����z�𒙂߂�
            auto grad = std::make_unique<EvalParams<LearnEvalType>>();
            std::array<double, 2> loss{ 0.0, 0.0 };
            for (const auto& data : replay_buffer_->makeBatch(BATCH_SIZE)) {
                pos.loadData(data.first);
                loss += addGrad(*grad, pos, data.second);
            }
            loss /= BATCH_SIZE;
            grad->forEach([this](CalcType& g) {
                g /= BATCH_SIZE;
            });

            //�w�K
            updateParams(*eval_params, *grad);

            //�w�K���̌���
            if (step_num == MAX_STEP_NUM / 2
                || step_num == MAX_STEP_NUM * 3 / 4) {
                LEARN_RATE *= 0.1;
            }

            //���Ԃ̌v��
            auto step_end = std::chrono::steady_clock::now();
            auto ela = std::chrono::duration_cast<std::chrono::milliseconds>(step_end - step_start);

            if (step_num % EVALUATION_INTERVAL == 0 || step_num == MAX_STEP_NUM) {
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
                
                //�]��
                evaluate();
                print("\n", false);

                //�����o��
                eval_params->writeFile("tmp" + std::to_string(i) + "_" + std::to_string(step_num) + ".bin");

            } else if (step_num % PRINT_INTERVAL == 0) {
                //�w�K���̕\������
                timestamp();
                print(step_num);
                print(POLICY_LOSS_COEFF * loss[0] + VALUE_LOSS_COEFF * loss[1]);
                print(loss[0]);
                print(loss[1]);
                print(LEARN_RATE * grad->maxAbs());
                print(LEARN_RATE * grad->sumAbs());
                print(eval_params->maxAbs());
                print(eval_params->sumAbs());
                print("\n", false);
            }

            //�w�K�ɂ����������Ԃ̒萔�{���邱�Ƃŋ^���I��Actor�̐��𑝂₷
            std::this_thread::sleep_for(ela * (WAIT_COEFF - 1));
        }

        log_file_.close();
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
        auto searcher = std::make_unique<Searcher>(usi_option.USI_Hash);

        Game game;
        Position pos(*eval_params);

        while (!pos.isFinish()) {
            auto move_and_teacher = searcher->think(pos, true);
            Move best_move = move_and_teacher.first;
            TeacherType teacher = move_and_teacher.second;

            pos.doMove(best_move);
            game.moves.push_back(best_move);
            game.teachers.push_back(teacher);
        }

        //�΋ǌ��ʂ̐ݒ�
        game.result = pos.resultForBlack();

        replay_buffer_->push(game);
    }
}

void AlphaZeroTrainer::evaluate() {
    replay_buffer_->mutex.lock();

    //�΋ǂ���p�����[�^������
    auto opponent_parameters_ = std::make_unique<EvalParams<DefaultEvalType>>();
    opponent_parameters_->readFile();

    //random_turn�͏����߂ɂ���
    auto copy = usi_option.random_turn;
    usi_option.random_turn = (uint32_t)EVALUATION_RANDOM_TURN;
    auto test_games = parallelPlay(*eval_params, *opponent_parameters_, EVALUATION_GAME_NUM, false);
    usi_option.random_turn = copy;

    //�o��
    for (int32_t i = 0; i < std::min(2, (int32_t)test_games.size()); i++) {
        test_games[i].writeKifuFile(EVAL_GAMES_DIR);
    }

    double win_rate = 0.0;
    for (int32_t i = 0; i < test_games.size(); i++) {
        win_rate += (i % 2 == 0 ? test_games[i].result : 1.0 - test_games[i].result);
    }

    //�d���̊m�F�����Ă݂�
    int32_t same_num = 0;
    for (int32_t i = 0; i < test_games.size(); i++) {
        for (int32_t j = i + 1; j < test_games.size(); j++) {
            if (test_games[i] == test_games[j]) {
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

    replay_buffer_->mutex.unlock();
}

std::vector<Game> AlphaZeroTrainer::parallelPlay(const EvalParams<DefaultEvalType>& curr, const EvalParams<DefaultEvalType>& target, int32_t game_num, bool add_noise) {
    std::vector<Game> games(game_num);
    std::atomic<int32_t> index;
    index = 0;

    std::vector<std::thread> threads;
    for (int32_t i = 0; i < (int32_t)usi_option.thread_num; i++) {
        threads.emplace_back([&]() {
            auto searcher = std::make_unique<Searcher>(usi_option.USI_Hash);
            while (true) {
                int32_t curr_index = index++;
                if (curr_index >= game_num) {
                    return;
                }
                Game& game = games[curr_index];
                game.moves.reserve(70);
                game.teachers.reserve(70);
                Position pos_c(curr), pos_t(target);

                while (!pos_c.isFinish()) {
                    //i�������̂Ƃ�pos_c�����
                    auto move_and_teacher = ((pos_c.turn_number() % 2) == (curr_index % 2) ?
                        searcher->think(pos_c, add_noise) :
                        searcher->think(pos_t, add_noise));
                    Move best_move = move_and_teacher.first;
                    TeacherType teacher = move_and_teacher.second;

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