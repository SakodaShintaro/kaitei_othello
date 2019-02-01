#ifndef REPLAY_BUFFER_HPP
#define REPLAY_BUFFER_HPP

#include<mutex>

#include"game.hpp"

class ReplayBuffer {
public:
    //�o�b�`�T�C�Y���̃f�[�^��Ԃ��֐�
    std::vector<std::pair<std::array<int64_t, 3>, TeacherType>> makeBatch(int64_t batch_size);

    //1�Ǖ��̃f�[�^���l�ߍ��ފ֐�
    void push(Game& game);

    //����������֐�
    void clear();

    //AlphaZeroTainer�N���X������A�N�Z�X�ł���悤��public�ɒu��
    //TDLeaf(��)�̃�
    double LAMBDA;

    //�X�^�b�N�T�C�Y�̏��
    int64_t MAX_STACK_SIZE;

    //�ŏ��ɑ҂�
    int64_t WAIT_LIMIT_SIZE;

    //�r������p
    std::mutex mutex;

private:
    //�f�[�^���i�[����o�b�t�@
    std::vector<std::pair<std::array<int64_t, 3>, TeacherType>> buffer_;
};

#endif