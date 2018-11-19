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

    //1�ǂ����������Ă�����J��Ԃ��w�K���Ă݂�e�X�g�p�̊֐�
    void testLearn();

private:
    //--------------------
    //    �������\�b�h
    //--------------------
    //���ȑ΋ǂ��s���֐�
    static std::vector<Game> play(int32_t game_num, int32_t search_limit, bool add_noise);

    //���񉻂��đ΋ǂ��s���֐�
    static std::vector<Game> parallelPlay(const EvalParams<DefaultEvalType>& curr, const EvalParams<DefaultEvalType>& target, int32_t game_num, int32_t search_limit, bool add_noise);

    //
    void learnSlave();

    //���t�@�C���ɕۑ�����Ă���p�����[�^�Ƒ΋ǂ��ċ����𑪒肷��֐�
    void evaluate();

    //�����Q���瑹���E���z�E����萔�E���萔�ɂ��������������v�Z����֐�
    std::array<double, 2> learnGames(const std::vector<Game>& games, EvalParams<LearnEvalType>& grad);

    //���������葤����Đ����đ����E���z���v�Z����֐�:elmo�i��ɑΉ�
    void learnOneGame(const Game& game, EvalParams<LearnEvalType>& grad, std::array<double, 2>& loss, uint64_t& learn_position_num);

    //�������ŏI�肩��Đ����đ����E���z���v�Z����֐�:Sarsa�ɑΉ�?
    void learnOneGameReverse(const Game& game, EvalParams<LearnEvalType>& grad, std::array<double, 2>& loss, uint64_t& learn_position_num);

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
    int32_t EVALUATION_INTERVAL;

    //�]������Ƃ��̃����_���萔
    int32_t EVALUATION_RANDOM_TURN;

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
    std::vector<std::pair<std::array<int64_t, 2>, TeacherType>> position_stack_;
    
    //�����Ȃ��Đ��オ�i�񂾉�
    uint64_t update_num_;

    //�����Ȃ����Ƃ݂Ȃ��Ȃ�������
    uint64_t fail_num_;

    //�����Ȃ��Ă��Ȃ����Ƃ������Ă����
    uint64_t consecutive_fail_num_;

    //���Ϗ���
    double win_average_;
};

#endif