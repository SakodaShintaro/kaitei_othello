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
    std::array<double, 2> addGrad(EvalParams<LearnEvalType>& grad, Position& pos, TeacherType teacher);

    //�t�`�d�������Ă��邩���l�����ƏƂ炵���킹�Č��؂���֐�
    void verifyAddGrad(Position& pos, TeacherType teacher);

    //���z�����ƂɃp�����[�^���X�V����֐�
    void updateParams(EvalParams<LearnEvalType>& params, const EvalParams<LearnEvalType>& grad);
    void updateParamsSGD(EvalParams<LearnEvalType>& params, const EvalParams<LearnEvalType>& grad);
    void updateParamsMomentum(EvalParams<LearnEvalType>& params, const EvalParams<LearnEvalType>& grad, EvalParams<LearnEvalType>& pre_update);

    //--------------------
    //    ���̑��֐���
    //--------------------
    //log_file_�Ɍo�ߎ��Ԃ��o�͂���֐�
    void timestamp();

    //�W���o�͂�log_file_�̗����ɏo�͂���֐�
    template<class T> void print(T t);

    //optimizer�Ƃ��ē��͂��ꂽ���̂����������肷��֐�
    bool isLegalOptimizer();

    //-----------------------------------------------------
    //    �t�@�C������ǂݍ��ނ���const���͂��Ă��Ȃ���
    //    �قڒ萔�ł������
    //-----------------------------------------------------
    //�w�K��
    double LEARN_RATE;

    //�w�K��������������Ƃ��̌W��
    double LEARN_RATE_DECAY;

    //Momentum�ɂ����鍬����
    double MOMENTUM_DECAY;

    //�o�b�`�T�C�Y
    int32_t BATCH_SIZE;

    //optimizer�̐ݒ�
    std::string OPTIMIZER_NAME;

    //���񉻂���X���b�h��
    uint32_t THREAD_NUM;

    //policy_loss�ɂ�����W��
    double POLICY_LOSS_COEFF;

    //value_loss�ɂ�����W��
    double VALUE_LOSS_COEFF;

    //--------------------------------
    //    �w�K���ɗp���郁���o�ϐ�
    //--------------------------------
    //���O�t�@�C��
    std::ofstream log_file_;

    //�w�K�J�n����
    std::chrono::time_point<std::chrono::steady_clock> start_time_;


    std::unique_ptr<EvalParams<LearnEvalType>> pre_update_;
};

template<class T>
inline void BaseTrainer::print(T t) {
    std::cout << t << "\t";
    log_file_ << t << "\t";
}

#endif // !TRAINER_HPP