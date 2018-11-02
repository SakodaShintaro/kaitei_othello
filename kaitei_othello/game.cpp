#include"game.hpp"
#include"position.hpp"
#include"network.hpp"
#include<iostream>
#include<algorithm>
#include<string>
#include<unordered_map>
#include<experimental/filesystem>
#include<atomic>

void Game::writeKifuFile(std::string dir_path) const {
    static std::atomic<int64_t> id;
    std::string file_name = dir_path + std::to_string(id++) + ".kif";
    std::ofstream ofs(file_name);
    if (!ofs) {
        std::cout << "cannnot open " << dir_path << std::endl;
        assert(false);
    }
    ofs << std::fixed;

    Position pos(*eval_params);

    for (int32_t i = 0; i < moves.size(); i++) {
        Move m = moves[i];
        ofs << i + 1 << " ";
        File to_file = SquareToFile[m.to()];
        Rank to_rank = SquareToRank[m.to()];
        ofs << fileToString[to_file] << rankToString[to_rank];
        ofs << "**対局 評価値 " << (i % 2 == 0 ? m.score : -m.score) << std::endl;

        if (i == 0) {
            auto p = pos.maskedPolicy();
            for (auto& move : pos.generateAllMoves()) {
                ofs << "*" << move << " " << teachers[i][move.toLabel()] << " " << p[move.toLabel()] << std::endl;
            }
        }

        ofs << "*valueForTurn = " << pos.valueScoreForTurn() << std::endl;
        ofs << "*sigmoid(val) = " << sigmoid(pos.valueScoreForTurn(), 1.0) << std::endl;

        pos.doMove(m);
    }
    
    ofs << pos.score() << std::endl;
}
