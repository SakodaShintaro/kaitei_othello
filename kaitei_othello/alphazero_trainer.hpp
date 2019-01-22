#pragma once

#ifndef ALPHAZERO_TRAINER_HPP
#define ALPHAZERO_TRAINER_HPP

#include"base_trainer.hpp"
#include"game.hpp"

class AlphaZeroTrainer : BaseTrainer {
public:
    //--------------------
    //    公開メソッド
    //--------------------
    //コンストラクタ
    AlphaZeroTrainer(std::string settings_file_path);

    //1スレッドだけ学習器を作り、残りのスレッドは自己対局
    //これらは並列に行われる
    void learn();

private:
    //--------------------
    //    内部メソッド
    //--------------------
    //棋譜生成を行う関数
    void learnSlave();

    //今ファイルに保存されているパラメータと対局して強さを測定する関数
    void evaluate();

    //TDLeaf(λ)で教師信号を計算しながら学習データプールに詰めていく関数
    void pushOneGame(Game& game);

    //---------------------------------------------
    //    ファイルから読み込むためconst化はして
    //    いないがほぼ定数であるもの
    //---------------------------------------------
    //TDLeaf(λ)のλ
    double LAMBDA;

    //強くなったとみなす勝率の閾値
    double THRESHOLD;

    //評価する際のゲーム数
    int32_t EVALUATION_GAME_NUM;

    //評価する間隔
    int64_t EVALUATION_INTERVAL;

    //評価するときのランダム手数
    int32_t EVALUATION_RANDOM_TURN;

    //スタックサイズの上限
    int64_t MAX_STACK_SIZE;

    //ステップ数
    int64_t MAX_STEP_NUM;

    //最初に待つ量
    int64_t WAIT_LIMIT_SIZE;

    //疑似的に学習時間を倍増させてActorの数を増やす係数
    double WAIT_COEFF;

    //------------
    //    変数
    //------------
    //学習用に加工済の局面と教師データのセットをプールするもの
    std::vector<std::pair<std::array<int64_t, 3>, TeacherType>> position_pool_;
    
    //強くなって世代が進んだ回数
    uint64_t update_num_;
};

#endif