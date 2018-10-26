#include"treestrap_trainer.hpp"
#include"rootstrap_trainer.hpp"
#include<thread>

TreestrapTrainer::TreestrapTrainer(std::string settings_file_path) {
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
                std::cerr << "Optimizer��[SGD, AdaGrad, RMSProp, AdaDelta]����I��" << std::endl;
                assert(false);
            }
        } else if (name == "learn_rate") {
            ifs >> LEARN_RATE;
        } else if (name == "thread_num") {
            ifs >> THREAD_NUM;
            THREAD_NUM = std::min(std::max(1u, THREAD_NUM), std::thread::hardware_concurrency());
        } else if (name == "random_turn") {
            ifs >> usi_option.random_turn;
        } else if (name == "random_move_temperature") {
            ifs >> usi_option.temperature;
        } else if (name == "step_size") {
            ifs >> step_size;
        } else {
            std::cerr << "Error! There is no such setting." << std::endl;
            assert(false);
        }
    }
}

void TreestrapTrainer::startLearn() {
    std::cout << "start treeStrap" << std::endl;
    eval_params->readFile();

    //�w�K�p�ɃI�v�V������ύX
    usi_option.draw_turn = 256;

    //���z�̏���
    grad_ = std::make_unique<EvalParams<LearnEvalType>>();

    //���O���o�͂���t�@�C���̏���
    log_file_.open("treestrap_log.txt");
    log_file_ << "step\tloss" << std::endl;
    log_file_ << std::fixed;
    std::cout << std::fixed;

    for (int32_t step = 0; step < step_size; step++) {
        grad_->clear();
        loss_ = 0.0;
        for (int32_t i = 0; i < BATCH_SIZE; i++) {
            Position pos(*eval_params);
            //�ŏ�����̓����_���ɑI��
            while (pos.turn_number() < usi_option.random_turn) {
                auto random_move = Searcher::randomChoice(pos);
                pos.doMove(random_move);
            }

            //��������Treestrap
            while (true) {
                auto moves = pos.generateAllMoves();
                Move best_move;
                best_move.score = MIN_SCORE;
                for (auto move : moves) {
                    pos.doMove(move);
                    //Score score = -miniMaxLearn(pos, SEARCH_DEPTH * PLY);
                    Score score = -alphaBetaLearn(pos, MIN_SCORE, best_move.score, SEARCH_DEPTH * PLY);
                    if (score > best_move.score) {
                        best_move = move;
                        best_move.score = score;
                    }
                    pos.undo();
                }

                if (isMatedScore(best_move.score)) {
                    break;
                }

                Score curr_score = pos.scoreForTurn();
                Score best_score = best_move.score;

                //���ǖʂ̊w�K
#ifdef USE_NN
                assert(false);
#else
                loss_ += calcLoss(curr_score, best_score);
                updateGradient(*grad_, pos.features(), (LearnEvalType)calcGrad(curr_score, best_score));
#endif
                learned_position_num_++;

                pos.doMove(best_move);

                Score dummy;
                if (pos.isRepeating(dummy) || pos.turn_number() >= 256) {
                    break;
                }
            }
        }

        //�X�V
        //updateParamsSGD(*eval_params, *grad_);
        eval_params->writeFile();
        std::cout  << "step = " << std::setw(4) << step << " loss = " << std::setw(10) << loss_ / learned_position_num_ << std::endl;
        log_file_  << "step = " << std::setw(4) << step << " loss = " << std::setw(10) << loss_ / learned_position_num_ << std::endl;
    }
    std::cout << "finish treeStrap" << std::endl;
}

Score TreestrapTrainer::miniMaxLearn(Position& pos, Depth depth) {
    //���ǖʂ̕]���l
    Score curr_score = pos.scoreForTurn();

    if (depth < PLY) {
        //�T���I��
        //�Î~�T���͓����ׂ��H
        return curr_score;
    }

    Score best_score = MIN_SCORE;
    auto moves = pos.generateAllMoves();

    for (auto move : moves) {
        pos.doMove(move);
        Score score = -miniMaxLearn(pos, depth - PLY);
        pos.undo();

        best_score = std::max(best_score, score);
    }

    //pos.print();
    //std::cout << "best_score = " << best_score << std::endl;
    //std::cout << "curr_score = " << curr_score << std::endl;

    if (isMatedScore(best_score)) {
        //�w�K����return
        return best_score;
    }

    //���ǖʂ̊w�K
#ifdef USE_NN
    assert(false);
#else
    loss_ += calcLoss(curr_score, best_score);
    updateGradient(*grad_, pos.features(), (LearnEvalType)calcGrad(curr_score, best_score));
#endif
    learned_position_num_++;

    return best_score;
}

Score TreestrapTrainer::alphaBetaLearn(Position& pos, Score alpha, Score beta, Depth depth) {
    //���ǖʂ̕]���l
    Score curr_score = pos.scoreForTurn();

    if (depth < PLY) {
        //�T���I��
        //�Î~�T���͓����ׂ��H
        return curr_score;
    }

    Score best_score = MIN_SCORE;
    auto moves = pos.generateAllMoves();

    for (auto move : moves) {
        pos.doMove(move);
        Score score = -alphaBetaLearn(pos, -beta, -alpha, depth - PLY);
        pos.undo();

        if (score > best_score) {
            best_score = score;
            if (best_score >= beta) {
                //beta-cut����
                //�w�K�́c�c���Ȃ��H
                return best_score;
            } else if (best_score > alpha) {
                alpha = best_score;
            }
        }
    }

    if (isMatedScore(best_score)) {
        //�w�K����return
        return best_score;
    }

    //���ǖʂ̊w�K
#ifdef USE_NN
    assert(false);
#else
    loss_ += calcLoss(curr_score, best_score);
    updateGradient(*grad_, pos.features(), (LearnEvalType)calcGrad(curr_score, best_score));
#endif
    learned_position_num_++;

    return best_score;
}

double TreestrapTrainer::calcLoss(Score shallow_score, Score deep_score) {
    double y = sigmoid((double)shallow_score, CP_GAIN);
    double t = sigmoid((double)deep_score, CP_GAIN);
    return binaryCrossEntropy(y, t);
}

double TreestrapTrainer::calcGrad(Score shallow_score, Score deep_score) {
    double y = sigmoid((double)shallow_score, CP_GAIN);
    double t = sigmoid((double)deep_score, CP_GAIN);
    return y - t;
}
