#pragma once

#ifndef LOAD_GAME_HPP
#define LOAD_GAME_HPP

#include"move.hpp"
#include"eval_params.hpp"
#include<string>
#include<vector>
#include<experimental/filesystem>

struct Game {
	//指し手のvector
	std::vector<Move> moves;

    //教師データのvector
    std::vector<TeacherType> teachers;
	
    //対局結果
    static constexpr double RESULT_BLACK_WIN = 1.0;
    static constexpr double RESULT_WHITE_WIN = 0.0;
    static constexpr double RESULT_DRAW = 0.5;
	double result;

    void writeKifuFile(std::string dir_path) const;
};

#endif // !LOAD_GAME_HPP