#pragma once

#ifndef TRAINER_HPP
#define TRAINER_HPP

#include"eval_params.hpp"
#include"position.hpp"
#include<iomanip>
#include<fstream>
#include<chrono>
#include<ctime>

//�eTrainer�̊��ƂȂ�N���X
class BaseTrainer {
protected:
    //------------------------------------
    //    �p�����[�^�X�V�Ɋւ���֐���
    //------------------------------------
    //���ǖʂɑ΂���]���֐��̏o�͂�teacher�ɋ߂Â���悤�Ɍ��z���X�V����֐�
#ifdef USE_NN
    std::array<double, 2> addGrad(EvalParams<LearnEvalType>& grad, Position& pos, TeacherType teacher);
#else
    double addGrad(EvalParams<LearnEvalType>& grad, Position& pos, TeacherType teacher);
#endif

#ifdef USE_NN
    //�t�`�d�������Ă��邩���l�����ƏƂ炵���킹�Č��؂���֐�
    void verifyAddGrad(Position& pos, TeacherType teacher);
#endif

    //���z�����ƂɃp�����[�^���X�V����֐�
    void updateParams(EvalParams<LearnEvalType>& params, const EvalParams<LearnEvalType>& grad);
    void updateParamsSGD(EvalParams<LearnEvalType>& params, const EvalParams<LearnEvalType>& grad);
    void updateParamsMomentum(EvalParams<LearnEvalType>& params, const EvalParams<LearnEvalType>& grad, EvalParams<LearnEvalType>& pre_update);

#ifndef USE_NN
    //KPPT�Ɋւ��ē������󂯎���Ă����p�����[�^�ɂ��Ă̌��z���X�V����֐�:Bonanza Method�Ŏg�����ߏ����Ȃ�
    void updateGradient(EvalParams<LearnEvalType>& grad, const Features &ee, const LearnEvalType delta);
#endif

    //--------------------
    //    ���̑��֐���
    //--------------------
    //log_file_�Ɍo�ߎ��Ԃ��o�͂���֐�
    void timestamp();

    //optimizer�Ƃ��ē��͂��ꂽ���̂����������肷��֐�
    bool isLegalOptimizer();

    //-----------------------------------------------------
    //    �t�@�C������ǂݍ��ނ���const���͂��Ă��Ȃ���
    //    �قڒ萔�ł������
    //-----------------------------------------------------
    //�w�K��
    double LEARN_RATE;

    //Momentum�ɂ����鍬����
    double MOMENTUM_DECAY;

    //�o�b�`�T�C�Y
    int32_t BATCH_SIZE;

    //�T���[��
    int32_t SEARCH_DEPTH;

    //optimizer�̐ݒ�
    std::string OPTIMIZER_NAME;

    //���񉻂���X���b�h��
    uint32_t THREAD_NUM;

#ifdef USE_NN
    //value_loss�ɂ�����W��
    double VALUE_COEFF;
#endif

    //--------------------------------
    //    �w�K���ɗp���郁���o�ϐ�
    //--------------------------------
    //���O�t�@�C��
    std::ofstream log_file_;

    //�w�K�J�n����
    std::chrono::time_point<std::chrono::steady_clock> start_time_;

    //�w�K���̃p�����[�^:KPPT�̂Ƃ������g��
#ifndef USE_NN
    std::unique_ptr<EvalParams<LearnEvalType>> learning_parameters;
#endif

    std::unique_ptr<EvalParams<LearnEvalType>> pre_update_;
};

#endif // !TRAINER_HPP