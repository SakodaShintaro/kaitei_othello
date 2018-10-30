#pragma once
#include"base_trainer.hpp"
#include"types.hpp"
#include"move.hpp"
#include"eval_params.hpp"
#include"searcher.hpp"
#include"game.hpp"
#include<vector>

class RootstrapTrainer : BaseTrainer {
public:
    //--------------------
    //    ���J���\�b�h
    //--------------------
    //�R���X�g���N�^
    RootstrapTrainer(std::string settings_file_path);

    //�e�X���b�h���񓯊��I�Ɋ��������ƃp�����[�^�X�V���s���֐�
    //learnSync�ƑI���ŕЕ����g��
    void learnAsync();

    //���������������񉻂��p�����[�^�X�V��1�X���b�h�ōs���֐�
    void learnSync();

    //1�ǂ����������Ă�����J��Ԃ��w�K���Ă݂�e�X�g�p�̊֐�
    void testLearn();

private:
    //--------------------
    //    �������\�b�h
    //--------------------
    //���������E�p�����[�^�X�V���s��1�X���b�h���̊֐�
    //����I�Ƀ`�F�b�N������
    void learnAsyncSlave(int32_t id);

    //���ȑ΋ǂ��s���֐�
    static std::vector<Game> play(int32_t game_num, int32_t search_limit);

    //���񉻂��đ΋ǂ��s���֐�
    static std::vector<Game> parallelPlay(const EvalParams<DefaultEvalType>& curr, const EvalParams<DefaultEvalType>& target, int32_t game_num, int32_t search_limit);

    //�������珟�����v�Z����֐�
    static double calcCurrWinRate(const std::vector<Game>& games);

    //���t�@�C���ɕۑ�����Ă���p�����[�^�Ƒ΋ǂ��ċ����𑪒肷��֐�
    void evaluate();

    //�����Q���瑹���E���z�E����萔�E���萔�ɂ��������������v�Z����֐�
    void learnGames(const std::vector<Game>& games, std::array<double, 2>& loss, EvalParams<LearnEvalType>& grad);

    //���������葤����Đ����đ����E���z���v�Z����֐�:elmo�i��ɑΉ�
    std::array<double, 2> learnOneGame(const Game& game, EvalParams<LearnEvalType>& grad);

    //�������ŏI�肩��Đ����đ����E���z���v�Z����֐�:Sarsa�ɑΉ�?
    std::array<double, 2> learnOneGameReverse(const Game& game, EvalParams<LearnEvalType>& grad);

    //�W���o�͂�log_file_�̗����ɏo�͂���֐�
    template<class T> void print(T t);

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

    //------------
    //    �ϐ�
    //------------
    //���ȑ΋ǂő���ɂȂ�1����O�̃p�����[�^
    std::unique_ptr<EvalParams<DefaultEvalType>> opponent_parameters_;

    //�w�K�����ǖʐ�
    uint64_t sum_learned_games_;

    //�����Ȃ��Đ��オ�i�񂾉�
    uint64_t update_num_;

    //�����Ȃ����Ƃ݂Ȃ��Ȃ�������
    uint64_t fail_num_;

    //�����Ȃ��Ă��Ȃ����Ƃ������Ă����
    uint64_t consecutive_fail_num_;

    //���Ϗ���
    double win_average_;
};

template<class T>
inline void RootstrapTrainer::print(T t) {
    std::cout << t << "\t";
    log_file_ << t << "\t";
}