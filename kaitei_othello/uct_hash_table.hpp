﻿#ifndef UCT_HASH_ENTRY_HPP
#define UCT_HASH_ENTRY_HPP

#include"piece.hpp"
#include"position.hpp"
#include"eval_params.hpp"

using Index = int32_t;

struct UctHashEntry {
    int32_t sum_N;
    int32_t moves_size;
    std::vector<Move> moves;
    std::vector<Index> child_indices;
    std::vector<int32_t> N;
    std::vector<CalcType> policy;
    ValueType value;
    std::vector<ValueType> W;
    bool evaled;

    //識別用データ
    //ハッシュ値だけでは衝突が発生するので手数も持つ
    int64_t hash;
    int16_t turn_number;

    uint16_t age;

#ifdef USE_CATEGORICAL
    UctHashEntry() :
        sum_N(0), moves_size(0),
        evaled(false), hash(0), turn_number(0), age(0) {}
#else
    UctHashEntry() :
        sum_N(0), moves_size(0), value(0.0),
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

    // 未使用のインデックスを探して返す(開番地法)
    Index searchEmptyIndex(int64_t hash, int16_t turn_number);

    Index findSameHashIndex(int64_t hash, int16_t turn_number);

    void saveUsedHash(Position& pos, Index index);

    // 現在の局面をルートとする局面以外を削除する
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

    // 未展開のノードのインデックス
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
