#pragma once
#ifndef BOOK_HPP
#define BOOK_HPP

#include"move.hpp"
#include"position.hpp"
#include<random>

//����ǖʂɑ΂����Վ�̃f�[�^
class BookEntry {
public:
    BookEntry() {
        best_move[BLACK] = best_move[WHITE] = NULL_MOVE;
    }

    Move pickBest(const Color c) {
        return best_move[c];
    }

    Move pickOneMove(const Color c) {
        std::random_device seed_gen;
        std::mt19937 engine(seed_gen());
        std::shuffle(other_moves_[c].begin(), other_moves_[c].end(), engine);
        return other_moves_[c][0];
    }

    void update(const Move move, const Color c) {
        if (best_move[c] == NULL_MOVE) {
            best_move[c] = move;
            return;
        }
        if (best_move[c] != move) {
            other_moves_[c].push_back(best_move[c]);
            best_move[c] = move;
        }
    }

private:
    //�����K�v���ӊO�ɓ��
    //�Ƃ肠����best_move�Ƃ���ȊO�ɂ܂��������Ă����g�݂ł���Ă݂悤��
    Move best_move[ColorNum];
    std::vector<Move> other_moves_[ColorNum];
};

class Book {
public:
    Move probe(const Position& pos, const bool do_pick_best) const;
    void makeBookByThink();
    void makeBookByGames();
    void readFromFile();
    void writeToFile() const;
private:
    //sfen��������󂯎���Ă��̋ǖʂ̒�Վ����Ԃ�
    std::unordered_map<std::string, BookEntry> book_;
};

#endif // !BOOK_HPP