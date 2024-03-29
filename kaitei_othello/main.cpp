﻿#include"usi.hpp"
#include"usi_options.hpp"
#include"test.hpp"
#include"bitboard.hpp"
#include"network.hpp"
#include"MCTSearcher.hpp"

std::unique_ptr<EvalParams<DefaultEvalType>> eval_params(new EvalParams<DefaultEvalType>);

SharedData shared_data;

int main()
{
    Position::initHashSeed();

    Bitboard::init();

	NBoardProtocol usi;
	usi.loop();
}