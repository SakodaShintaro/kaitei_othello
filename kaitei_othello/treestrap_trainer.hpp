#pragma once
#include"base_trainer.hpp"
#include"position.hpp"

#ifndef USE_MCTS

class TreestrapTrainer : BaseTrainer {
public:
    TreestrapTrainer(std::string settings_file_path);
    void startLearn();
private:
    //�w�K���Ȃ���T������֐�
    Score miniMaxLearn(Position& pos, Depth depth);
    Score alphaBetaLearn(Position& pos, Score alpha, Score beta, Depth depth);

    //���b�p�[
    double calcLoss(Score shallow_score, Score deep_score);
    double calcGrad(Score shallow_score, Score deep_score);

    //------------------------------
    //    �w�K���ɒl���ς��ϐ�
    //------------------------------
    //����
    double loss_;

    //���z
    std::unique_ptr<EvalParams<LearnEvalType>> grad_;

    //�w�K�ǖʐ�
    uint64_t learned_position_num_;

    //----------------------------------
    //    �w�K���ɒl���ς��Ȃ��ϐ�
    //----------------------------------
    //�X�e�b�v��
    int32_t step_size;
};

#endif