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

    //1局だけ生成してそれを繰り返し学習してみるテスト用の関数
    void testLearn();

private:
    //--------------------
    //    内部メソッド
    //--------------------
    //自己対局を行う関数
    static std::vector<Game> play(int32_t game_num, int32_t search_limit, bool add_noise);

    //並列化して対局を行う関数
    static std::vector<Game> parallelPlay(const EvalParams<DefaultEvalType>& curr, const EvalParams<DefaultEvalType>& target, int32_t game_num, int32_t search_limit, bool add_noise);

    //
    void learnSlave();

    //今ファイルに保存されているパラメータと対局して強さを測定する関数
    void evaluate();

    //棋譜群から損失・勾配・千日手数・長手数による引き分け数を計算する関数
    std::array<double, 2> learnGames(const std::vector<Game>& games, EvalParams<LearnEvalType>& grad);

    //棋譜を初手側から再生して損失・勾配を計算する関数:elmo絞りに対応
    void learnOneGame(const Game& game, EvalParams<LearnEvalType>& grad, std::array<double, 2>& loss, uint64_t& learn_position_num);

    //棋譜を最終手から再生して損失・勾配を計算する関数:Sarsaに対応?
    void learnOneGameReverse(const Game& game, EvalParams<LearnEvalType>& grad, std::array<double, 2>& loss, uint64_t& learn_position_num);

    //---------------------------------------------
    //    ファイルから読み込むためconst化はして
    //    いないがほぼ定数であるもの
    //---------------------------------------------
    //損失の計算方法
    enum {
        ELMO_LEARN,  //elmo絞り     : 最終的な勝敗と現局面の深い探索の値を混合して教師信号とする
        N_STEP_SARSA //n-step SARSA : 現局面から終局までの評価値を減衰させながら利用
    };
    int32_t LEARN_MODE;

    //減衰係数
    double DECAY_RATE;

    //引き分けの対局も学習するか
    bool USE_DRAW_GAME;

    //強くなったとみなす勝率の閾値
    double THRESHOLD;

    //深い探索にかける係数
    double DEEP_COEFFICIENT;

    //評価する際のゲーム数
    int32_t EVALUATION_GAME_NUM;

    //評価する間隔
    int32_t EVALUATION_INTERVAL;

    //評価するときのランダム手数
    int32_t EVALUATION_RANDOM_TURN;

    //スタックサイズの上限
    int64_t MAX_STACK_SIZE;

    //ステップ数
    int64_t MAX_STEP_NUM;

    //------------
    //    変数
    //------------
    //自己対局で相手になる1世代前のパラメータ
    //これ実は要らないのかもしれない
    //std::unique_ptr<EvalParams<DefaultEvalType>> opponent_parameters_;

    //学習した局面数
    //これはstep_num * BATCH_SIZEに等しいはず
    //uint64_t sum_learned_games_;

    //学習用に加工済の局面スタック
    std::vector<std::pair<std::array<int64_t, 3>, TeacherType>> position_stack_;
    
    //強くなって世代が進んだ回数
    uint64_t update_num_;

    //強くなったとみなせなかった回数
    uint64_t fail_num_;

    //強くなっていないことが続いている回数
    uint64_t consecutive_fail_num_;

    //平均勝率
    double win_average_;
};

#endif