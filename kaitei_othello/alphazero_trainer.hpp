#ifndef ALPHAZERO_TRAINER_HPP
#define ALPHAZERO_TRAINER_HPP

#include<mutex>

#include"base_trainer.hpp"
#include"replay_buffer.hpp"

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

    //���񉻂��đ΋ǂ��s���֐�
    static std::vector<Game> parallelPlay(const EvalParams<DefaultEvalType>& curr, const EvalParams<DefaultEvalType>& target, int32_t game_num, bool add_noise);

private:
    //--------------------
    //    �������\�b�h
    //--------------------
    //�����������s���֐�
    void learnSlave();

    //���t�@�C���ɕۑ�����Ă���p�����[�^�Ƒ΋ǂ��ċ����𑪒肷��֐�
    void evaluate();

    //---------------------------------------------
    //    �t�@�C������ǂݍ��ނ���const���͂���
    //    ���Ȃ����قڒ萔�ł������
    //---------------------------------------------
    //�����Ȃ����Ƃ݂Ȃ�������臒l
    double THRESHOLD;

    //�]������ۂ̃Q�[����
    int32_t EVALUATION_GAME_NUM;

    //�]������Ԋu
    int64_t EVALUATION_INTERVAL;

    //�]������Ƃ��̃����_���萔
    int32_t EVALUATION_RANDOM_TURN;

    //�X�e�b�v��
    int64_t MAX_STEP_NUM;

    //�^���I�Ɋw�K���Ԃ�{��������Actor�̐��𑝂₷�W��
    double WAIT_COEFF;

    //�w�K�����s����Z�b�g��
    int64_t LEARN_NUM;

    //�w�K����\������Ԋu
    int64_t PRINT_INTERVAL;

    //------------
    //    �ϐ�
    //------------
    //���v���C�o�b�t�@��Inner Class�Ƃ��Đ錾����
    std::unique_ptr<ReplayBuffer> replay_buffer_;
    
    //�����Ȃ��Đ��オ�i�񂾉�
    uint64_t update_num_;
};

#endif