#pragma once

#ifndef TRAINER_HPP
#define TRAINER_HPP

#include"eval_params.hpp"
#include"position.hpp"
#include<iomanip>
#include<fstream>
#include<chrono>
#include<ctime>

//各Trainerの基底となるクラス
class BaseTrainer {
protected:
    //------------------------------------
    //    パラメータ更新に関する関数類
    //------------------------------------
    //現局面に対する評価関数の出力をteacherに近づけるように勾配を更新する関数
#ifdef USE_NN
    std::array<double, 2> addGrad(EvalParams<LearnEvalType>& grad, Position& pos, TeacherType teacher);
#else
    double addGrad(EvalParams<LearnEvalType>& grad, Position& pos, TeacherType teacher);
#endif

#ifdef USE_NN
    //逆伝播が合っているか数値微分と照らし合わせて検証する関数
    void verifyAddGrad(Position& pos, TeacherType teacher);
#endif

    //勾配をもとにパラメータを更新する関数
    void updateParams(EvalParams<LearnEvalType>& params, const EvalParams<LearnEvalType>& grad);
    void updateParamsSGD(EvalParams<LearnEvalType>& params, const EvalParams<LearnEvalType>& grad);
    void updateParamsMomentum(EvalParams<LearnEvalType>& params, const EvalParams<LearnEvalType>& grad, EvalParams<LearnEvalType>& pre_update);

#ifndef USE_NN
    //KPPTに関して特徴を受け取ってそれらパラメータについての勾配を更新する関数:Bonanza Methodで使うため消せない
    void updateGradient(EvalParams<LearnEvalType>& grad, const Features &ee, const LearnEvalType delta);
#endif

    //--------------------
    //    その他関数類
    //--------------------
    //log_file_に経過時間を出力する関数
    void timestamp();

    //optimizerとして入力されたものが正当か判定する関数
    bool isLegalOptimizer();

    //-----------------------------------------------------
    //    ファイルから読み込むためconst化はしていないが
    //    ほぼ定数であるもの
    //-----------------------------------------------------
    //学習率
    double LEARN_RATE;

    //Momentumにおける混合比
    double MOMENTUM_DECAY;

    //バッチサイズ
    int32_t BATCH_SIZE;

    //探索深さ
    int32_t SEARCH_DEPTH;

    //optimizerの設定
    std::string OPTIMIZER_NAME;

    //並列化するスレッド数
    uint32_t THREAD_NUM;

#ifdef USE_NN
    //value_lossにかける係数
    double VALUE_COEFF;
#endif

    //--------------------------------
    //    学習中に用いるメンバ変数
    //--------------------------------
    //ログファイル
    std::ofstream log_file_;

    //学習開始時間
    std::chrono::time_point<std::chrono::steady_clock> start_time_;

    //学習中のパラメータ:KPPTのときだけ使う
#ifndef USE_NN
    std::unique_ptr<EvalParams<LearnEvalType>> learning_parameters;
#endif

    std::unique_ptr<EvalParams<LearnEvalType>> pre_update_;
};

#endif // !TRAINER_HPP