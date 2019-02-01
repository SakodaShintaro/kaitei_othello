#ifndef REPLAY_BUFFER_HPP
#define REPLAY_BUFFER_HPP

#include<mutex>

#include"game.hpp"

class ReplayBuffer {
public:
    //バッチサイズ分のデータを返す関数
    std::vector<std::pair<std::array<int64_t, 3>, TeacherType>> makeBatch(int64_t batch_size);

    //1局分のデータを詰め込む関数
    void push(Game& game);

    //初期化する関数
    void clear();

    //AlphaZeroTainerクラスからもアクセスできるようにpublicに置く
    //TDLeaf(λ)のλ
    double LAMBDA;

    //スタックサイズの上限
    int64_t MAX_STACK_SIZE;

    //最初に待つ量
    int64_t WAIT_LIMIT_SIZE;

    //排他制御用
    std::mutex mutex;

private:
    //データを格納するバッファ
    std::vector<std::pair<std::array<int64_t, 3>, TeacherType>> buffer_;
};

#endif