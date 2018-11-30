#pragma once
#ifndef USI_OPTIONS_HPP
#define USI_OPTIONS_HPP

#include"MCTSearcher.hpp"

class USIOption{
public:
	int64_t random_turn;
    int64_t USI_Hash;
    int64_t thread_num;
    double temperature;

#ifdef USE_MCTS
    int64_t playout_limit;
#endif
};

extern USIOption usi_option;

#endif