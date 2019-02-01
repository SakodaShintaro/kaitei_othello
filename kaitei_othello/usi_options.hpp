#pragma once
#ifndef USI_OPTIONS_HPP
#define USI_OPTIONS_HPP

#include"MCTSearcher.hpp"

class USIOption{
public:
	int64_t random_turn;
    int64_t USI_Hash;
    int64_t thread_num;

#ifdef USE_MCTS
    int64_t playout_limit;
#else
    int32_t depth_limit;
    int64_t node_limit;
#endif
};

extern USIOption usi_option;

#endif