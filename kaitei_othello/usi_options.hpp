#pragma once
#ifndef USI_OPTIONS_HPP
#define USI_OPTIONS_HPP

#include"MCTSearcher.hpp"

class USIOption{
public:
	int64_t byoyomi_margin;
	uint32_t random_turn;
    uint64_t USI_Hash;
    uint32_t thread_num;
    double temperature;

#ifdef USE_MCTS
    uint64_t playout_limit;
#endif
};

extern USIOption usi_option;

#endif