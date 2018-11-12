#pragma once

#ifndef UCT_HASH_ENTRY_HPP
#define UCT_HASH_ENTRY_HPP

#include"piece.hpp"
#include"position.hpp"
#include"eval_params.hpp"

using Index = int32_t;

struct UctHashEntry {
    int32_t move_count;
    int32_t child_num;
    std::vector<Move> legal_moves;
    std::vector<Index> child_indices;
    std::vector<int32_t> child_move_counts;
    std::vector<CalcType> nn_rates;
#ifdef USE_CATEGORICAL
    std::array<CalcType, BIN_SIZE> value_dist;
    std::array<CalcType, BIN_SIZE> win_sum;
    std::vector<std::array<CalcType, BIN_SIZE>> child_wins;
#else
    CalcType value_win;
    CalcType win_sum;
    std::vector<CalcType> child_wins;
#endif
    bool evaled;

    //���ʗp�f�[�^
    //�n�b�V���l�����ł͏Փ˂���������̂Ŏ萔������
    int64_t hash;
    int16_t turn_number;

    uint16_t age;

#ifdef USE_CATEGORICAL
    UctHashEntry() :
        move_count(0), child_num(0),
        evaled(false), hash(0), turn_number(0), age(0) {}
#else
    UctHashEntry() :
        move_count(0), win_sum(0.0), child_num(0), value_win(0.0),
        evaled(false), hash(0), turn_number(0), age(0) {}
#endif
};

class UctHashTable {
public:
    UctHashTable(int64_t hash_size);

    UctHashEntry& operator[](Index i) {
        return table_[i];
    }

    const UctHashEntry& operator[](Index i) const {
        return table_[i];
    }

    void setSize(int64_t megabytes);

    // ���g�p�̃C���f�b�N�X��T���ĕԂ�(�J�Ԓn�@)
    Index searchEmptyIndex(int64_t hash, int16_t turn_number);

    Index findSameHashIndex(int64_t hash, int16_t turn_number);

    void saveUsedHash(Position& pos, Index index);

    // ���݂̋ǖʂ����[�g�Ƃ���ǖʈȊO���폜����
    void deleteOldHash(Position& root, bool leave_root);

    double getUsageRate() const {
        return (double)used_ / size_;
    }

    bool hasEnoughSize() {
        return enough_size_;
    }

    bool alreadyInit() {
        return table_.size() > 0;
    }

    int64_t size() {
        return size_;
    }

    // ���W�J�̃m�[�h�̃C���f�b�N�X
    static constexpr Index NOT_EXPANDED = -1;

private:
    Index hashToIndex(int64_t hash) {
        return ((hash & 0xffffffff) ^ ((hash >> 32) & 0xffffffff)) & (size_ - 1);
    }

    int64_t uct_hash_limit_;
    int64_t size_;
    int64_t used_;
    bool enough_size_;
    uint16_t age_;
    std::vector<UctHashEntry> table_;
};

#endif
