#ifndef ALPHAZERO_TRAINER_HPP
#define ALPHAZERO_TRAINER_HPP

#include<mutex>

#include"base_trainer.hpp"
#include"replay_buffer.hpp"

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

    //並列化して対局を行う関数
    static std::vector<Game> parallelPlay(const EvalParams<DefaultEvalType>& curr, const EvalParams<DefaultEvalType>& target, int32_t game_num, bool add_noise);

private:
    //--------------------
    //    内部メソッド
    //--------------------
    //棋譜生成を行う関数
    void learnSlave();

    //今ファイルに保存されているパラメータと対局して強さを測定する関数
    void evaluate();

    //---------------------------------------------
    //    ファイルから読み込むためconst化はして
    //    いないがほぼ定数であるもの
    //---------------------------------------------
    //強くなったとみなす勝率の閾値
    double THRESHOLD;

    //評価する際のゲーム数
    int32_t EVALUATION_GAME_NUM;

    //評価する間隔
    int64_t EVALUATION_INTERVAL;

    //評価するときのランダム手数
    int32_t EVALUATION_RANDOM_TURN;

    //ステップ数
    int64_t MAX_STEP_NUM;

    //疑似的に学習時間を倍増させてActorの数を増やす係数
    double WAIT_COEFF;

    //学習を試行するセット数
    int64_t LEARN_NUM;

    //学習情報を表示する間隔
    int64_t PRINT_INTERVAL;

    //------------
    //    変数
    //------------
    //リプレイバッファをInner Classとして宣言する
    std::unique_ptr<ReplayBuffer> replay_buffer_;
    
    //強くなって世代が進んだ回数
    uint64_t update_num_;
};

#endif