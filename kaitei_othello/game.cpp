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
    std::string file_name = dir_path + std::to_string(id++) + ".okif";
    std::ofstream ofs(file_name);
    if (!ofs) {
        std::cout << "cannot open " << dir_path << std::endl;
        assert(false);
    }
    ofs << std::fixed;

    Position pos(*eval_params);

    for (int32_t i = 0; i < moves.size(); i++) {
        Move m = moves[i];
        ofs << i + 1 << " ";
        File to_file = SquareToFile[m.to()];
        Rank to_rank = SquareToRank[m.to()];
        ofs << fileToString[to_file] << rankToString[to_rank] << std::endl;

        if (m == NULL_MOVE) {
            pos.doMove(m);
            continue;
        }

        ofs << "scoreForBlack = " << (i % 2 == 0 ? m.score : 1.0 - m.score) << std::endl;
        ofs << "valueForBlack = " << pos.valueForBlack() << std::endl;

        pos.doMove(m);
    }
    
    ofs << pos.scoreForBlack() << std::endl;
}

bool Game::operator==(const Game& rhs) const {
    if (moves.size() != rhs.moves.size()) {
        //手数が違えば不一致
        return false;
    }

    bool result = true;
    for (int32_t i = 0; i < moves.size(); i++) {
        if (moves[i] != rhs.moves[i]) {
            //指し手が違えば不一致
            result = false;
            break;
        }
    }

    return result;
}