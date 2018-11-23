#pragma once

#ifndef ALPHAZERO_TRAINER_HPP
#define ALPHAZERO_TRAINER_HPP

#include"base_trainer.hpp"
#include"game.hpp"

class AlphaZeroTrainer : BaseTrainer {
public:
    //--------------------
    //    ���J���\�b�h
    //--------------------
    //�R���X�g���N�^
    AlphaZeroTrainer(std::string settings_file_path);

    //1�X���b�h�����w�K������A�c��̃X���b�h�͎��ȑ΋�
    //�����͕���ɍs����
    void learn();

private:
    //--------------------
    //    �������\�b�h
    //--------------------
    //�����������s���֐�
    void learnSlave();

    //���t�@�C���ɕۑ�����Ă���p�����[�^�Ƒ΋ǂ��ċ����𑪒肷��֐�
    void evaluate();

    //���s�ƒT�����ʂ��������ċ��t�Ƃ���֐�:elmo�i��ɑΉ�
    void pushOneGame(Game& game);

    //�w�������������Ȃ���n�X�e�b�v��̒T���̒l�����t�Ƃ���֐�
    void pushOneGameReverse(Game& game);

    //---------------------------------------------
    //    �t�@�C������ǂݍ��ނ���const���͂���
    //    ���Ȃ����قڒ萔�ł������
    //---------------------------------------------
    //�����̌v�Z���@
    enum {
        ELMO_LEARN,  //elmo�i��     : �ŏI�I�ȏ��s�ƌ��ǖʂ̐[���T���̒l���������ċ��t�M���Ƃ���
        N_STEP_SARSA //n-step SARSA : ���ǖʂ���I�ǂ܂ł̕]���l�����������Ȃ��痘�p
    };
    int32_t LEARN_MODE;

    //�����W��
    double DECAY_RATE;

    //���������̑΋ǂ��w�K���邩
    bool USE_DRAW_GAME;

    //�����Ȃ����Ƃ݂Ȃ�������臒l
    double THRESHOLD;

    //�[���T���ɂ�����W��
    double DEEP_COEFFICIENT;

    //�]������ۂ̃Q�[����
    int32_t EVALUATION_GAME_NUM;

    //�]������Ԋu
    double EVALUATION_INTERVAL_EXP;

    //�]������Ƃ��̃����_���萔
    int32_t EVALUATION_RANDOM_TURN;

    //�X�^�b�N�T�C�Y�̏��
    int64_t MAX_STACK_SIZE;

    //�X�e�b�v��
    int64_t MAX_STEP_NUM;

    //------------
    //    �ϐ�
    //------------
    //���ȑ΋ǂő���ɂȂ�1����O�̃p�����[�^
    //������͗v��Ȃ��̂�������Ȃ�
    //std::unique_ptr<EvalParams<DefaultEvalType>> opponent_parameters_;

    //�w�K�����ǖʐ�
    //�����step_num * BATCH_SIZE�ɓ������͂�
    //uint64_t sum_learned_games_;

    //�w�K�p�ɉ��H�ς̋ǖʃX�^�b�N
    std::vector<std::pair<std::array<int64_t, 3>, TeacherType>> position_stack_;
    
    //�����Ȃ��Đ��オ�i�񂾉�
    uint64_t update_num_;

    //�����Ȃ����Ƃ݂Ȃ��Ȃ�������
    uint64_t fail_num_;

    //�����Ȃ��Ă��Ȃ����Ƃ������Ă����
    uint64_t consecutive_fail_num_;
};

#endif