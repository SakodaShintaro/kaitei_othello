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

    //TDLeaf(��)�ŋ��t�M�����v�Z���Ȃ���w�K�f�[�^�v�[���ɋl�߂Ă����֐�
    void pushOneGame(Game& game);

    //---------------------------------------------
    //    �t�@�C������ǂݍ��ނ���const���͂���
    //    ���Ȃ����قڒ萔�ł������
    //---------------------------------------------
    //TDLeaf(��)�̃�
    double LAMBDA;

    //�����Ȃ����Ƃ݂Ȃ�������臒l
    double THRESHOLD;

    //�]������ۂ̃Q�[����
    int32_t EVALUATION_GAME_NUM;

    //�]������Ԋu
    int64_t EVALUATION_INTERVAL;

    //�]������Ƃ��̃����_���萔
    int32_t EVALUATION_RANDOM_TURN;

    //�X�^�b�N�T�C�Y�̏��
    int64_t MAX_STACK_SIZE;

    //�X�e�b�v��
    int64_t MAX_STEP_NUM;

    //�ŏ��ɑ҂�
    int64_t WAIT_LIMIT_SIZE;

    //�^���I�Ɋw�K���Ԃ�{��������Actor�̐��𑝂₷�W��
    double WAIT_COEFF;

    //------------
    //    �ϐ�
    //------------
    //�w�K�p�ɉ��H�ς̋ǖʂƋ��t�f�[�^�̃Z�b�g���v�[���������
    std::vector<std::pair<std::array<int64_t, 3>, TeacherType>> position_pool_;
    
    //�����Ȃ��Đ��オ�i�񂾉�
    uint64_t update_num_;
};

#endif