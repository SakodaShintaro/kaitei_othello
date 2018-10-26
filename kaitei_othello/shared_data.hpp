#pragma once
#ifndef SHARED_DATA_HPP
#define SHARED_DATA_HPP

#include"hash_table.hpp"
#include<atomic>

struct SharedData {
    SharedData() : root(*eval_params) {}
    std::atomic<bool> stop_signal;
    HashTable hash_table;
    Position root;
    int64_t limit_msec;
};

extern SharedData shared_data;

#endif // !SHARED_DATA_HPP