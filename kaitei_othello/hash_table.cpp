#include "hash_table.hpp"
#include "common.hpp"
#include <iostream>

HashEntry* HashTable::find(int64_t key) {
	if (table_[key & key_mask_].flag_ == false
        || table_[key & key_mask_].hash_val != key) return nullptr;
	return &table_[key & key_mask_];
}

void HashTable::save(int64_t key, Move move, Score score, Depth depth, std::vector<Move> sorted_moves) {
    HashEntry* target = &table_[key & key_mask_];
    target->hash_val = key;
    target->best_move = move;
    target->depth = depth;
    target->best_move.score = score;
    target->sorted_moves = sorted_moves;
    if (!target->flag_) {
        target->flag_ = true;
        hashfull_++;
    }
}

void HashTable::save(int64_t key, Move move, Score score, Depth depth) {
	HashEntry* target = &table_[key & key_mask_];
	target->hash_val = key;
	target->best_move = move;
	target->depth = depth;
    target->best_move.score = score;
	if (!target->flag_) {
		target->flag_ = true;
		hashfull_++;
    }
}

void HashTable::setSize(int64_t megabytes) {
    int64_t bytes = megabytes * 1024 * 1024;
    size_ = (megabytes > 0 ? (1ull << MSB64(bytes / sizeof(HashEntry))) : 0);
    age_ = 0;
    key_mask_ = size_ - 1;
    hashfull_ = 0;
	table_.resize(size_);
}