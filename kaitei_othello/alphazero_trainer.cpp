#pragma once

#include"alphazero_trainer.hpp"
#include"rootstrap_trainer.hpp"
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

static std::mutex MUTEX;

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
        } else if (name == "random_move_num") {
            ifs >> usi_option.random_turn;
        } else if (name == "lambda") {
            ifs >> LAMBDA;
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
            position_pool_.reserve(MAX_STACK_SIZE);
        } else if (name == "max_step_num") {
            ifs >> MAX_STEP_NUM;
        } else if (name == "wait_limit_size") {
            ifs >> WAIT_LIMIT_SIZE;
        } else if (name == "wait_coeff"){
            ifs >> WAIT_COEFF;
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

    //�w�K
    for (int32_t i = 1; i <= 5 ; i++) {
        //���Ԃ�������
        start_time_ = std::chrono::steady_clock::now();

        MUTEX.lock();

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
        std::cout << std::endl;
        log_file_ << std::endl;

        position_pool_.clear();
        position_pool_.reserve(MAX_STACK_SIZE);

        MUTEX.unlock();

        for (int32_t step_num = 1; step_num <= MAX_STEP_NUM; step_num++) {
            //���̃X�e�b�v�ɂ����������Ԃ�������
            auto step_start = std::chrono::steady_clock::now();

            //�~�j�o�b�`�����z�𒙂߂�
            auto grad = std::make_unique<EvalParams<LearnEvalType>>();
            std::array<double, 2> loss{ 0.0, 0.0 };
            for (int32_t j = 0; j < BATCH_SIZE; j++) {
                if ((int64_t)position_pool_.size() <= BATCH_SIZE * WAIT_LIMIT_SIZE) {
                    j--;
                    continue;
                }

                //�����_���ɑI��
                MUTEX.lock();
                int32_t random = engine() % position_pool_.size();
                auto data = position_pool_[random];
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

            //�w�K���̌���
            LEARN_RATE *= LEARN_RATE_DECAY;

            //���Ԃ̌v��
            auto step_end = std::chrono::steady_clock::now();
            auto ela = std::chrono::duration_cast<std::chrono::milliseconds>(step_end - step_start);

            //�]���Ə����o��
            if (step_num % EVALUATION_INTERVAL == 0 || step_num == MAX_STEP_NUM) {
                evaluate();
                eval_params->writeFile("tmp" + std::to_string(i) + "_" + std::to_string(step_num) + ".bin");
            }

            std::cout << std::endl;
            log_file_ << std::endl;

            MUTEX.unlock();

            //�w�K�ɂ����������Ԃ̒萔�{���邱�Ƃŋ^���I��Actor�̐��𑝂₷
            std::this_thread::sleep_for(ela * WAIT_COEFF);
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
        auto games = RootstrapTrainer::play(1, true);

        MUTEX.lock();
        //���������������w�K�p�f�[�^�ɉ��H����stack�֋l�ߍ���
        for (auto& game : games) {
            pushOneGame(game);
        }

        if ((int64_t)position_pool_.size() >= MAX_STACK_SIZE) {
            auto diff = position_pool_.size() - MAX_STACK_SIZE;
            position_pool_.erase(position_pool_.begin(), position_pool_.begin() + diff);
        }
        MUTEX.unlock();
    }
}

void AlphaZeroTrainer::evaluate() {
    //�΋ǂ���p�����[�^������
    auto opponent_parameters_ = std::make_unique<EvalParams<DefaultEvalType>>();
    opponent_parameters_->readFile();

    //random_turn�͏����߂ɂ���
    auto copy = usi_option.random_turn;
    usi_option.random_turn = (uint32_t)EVALUATION_RANDOM_TURN;
    auto test_games = RootstrapTrainer::parallelPlay(*eval_params, *opponent_parameters_, EVALUATION_GAME_NUM, false);
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
}

void AlphaZeroTrainer::pushOneGame(Game& game) {
    Position pos(*eval_params);

    //�܂��͍ŏI�ǖʂ܂œ�����
    for (auto move : game.moves) {
        pos.doMove(move);
    }

    assert(Game::RESULT_WHITE_WIN <= game.result && game.result <= Game::RESULT_BLACK_WIN);

    //��肩�猩������,���z.�w���ړ����ςœ������Ă���.�ŏ��͌��ʂɂ���ď�����(0 or 0.5 or 1)
    double win_rate_for_black = game.result;

    for (int32_t i = (int32_t)game.moves.size() - 1; i >= 0; i--) {
        //i�Ԗڂ̎w���肪�Ή�����̂�1��߂����ǖ�
        pos.undo();

        if (game.moves[i] == NULL_MOVE) { 
            //�p�X��������w�K���΂�
            continue;
        }

#ifdef USE_CATEGORICAL
        //��Ԃ��猩�����z�𓾂�
        auto teacher_dist = onehotDist(pos.color() == BLACK ? win_rate_for_black : 1.0 - win_rate_for_black);

        //teacher�ɃR�s�[����
        std::copy(teacher_dist.begin(), teacher_dist.end(), game.teachers[i].begin() + POLICY_DIM);
#else
        //teacher�ɃR�s�[����
        game.teachers[i][POLICY_DIM] = (CalcType)(pos.color() == BLACK ? win_rate_for_black : 1.0 - win_rate_for_black);
#endif
        //�T�����ʂ��肩�猩���l�ɕϊ�
        double curr_win_rate = (pos.color() == BLACK ? game.moves[i].score : 1.0 - game.moves[i].score);

        //����
        win_rate_for_black = LAMBDA * win_rate_for_black + (1.0 - LAMBDA) * curr_win_rate;

        //�X�^�b�N�ɋl�߂�
        position_pool_.push_back({ pos.data(), game.teachers[i] });
    }
}