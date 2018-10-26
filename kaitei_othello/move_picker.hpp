#pragma once

#include"move.hpp"
#include<vector>

//enum Depth;
class Position;
class History;

class MovePicker {
private:
	Move* begin() { return moves_; }
    Move* end() { return end_; }

	Position& pos_;

	Move killer_moves_[2];
	Move counter_move_;
	Depth depth;
	Move tt_move_;

	//ProbCut�p�̎w���萶���ɗp����A���O�̎w����ŕߊl���ꂽ��̉��l
	//int threshold;

	//�w���萶���̒i�K
	int stage_;

	//���ɕԂ���A�������ꂽ�w����̖����ABadCapture�̏I�[
    Move *cur_, *end_;
    Move *bad_capture_start_, *bad_capture_end_;

    Move *moves_;

    const History& history_;

public:
	//�ʏ�T������Ă΂��ۂ̃R���X�g���N�^
#ifdef USE_SEARCH_STACK
	MovePicker(Position& pos, const Move ttMove, const Depth depth, const History& history, const Move killers[2], Move counter_move);
#else
    MovePicker(Position& pos, const Move ttMove, const Depth depth, const History& history, Move counter_move);
#endif

    //�Î~�T������Ă΂��ۂ̃R���X�g���N�^
    MovePicker(Position& pos, const Move ttMove, const Depth depth, const History& history);

    ~MovePicker() {
        delete[] moves_;
    }
	Move nextMove();
	void generateNextStage();
    void scoreCapture();
    void scoringWithHistory();

    int stage() {
        return stage_;
    }

    void printAllMoves() {
        for (auto itr = cur_; itr != end_; itr++) {
            itr->printWithScore();
        }
    }
};