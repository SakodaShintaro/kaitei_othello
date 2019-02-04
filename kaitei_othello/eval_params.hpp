#pragma once
#ifndef EVAL_PARAMS_HPP
#define EVAL_PARAMS_HPP

#include"piece.hpp"
#include"square.hpp"
#include<climits>
#include<cmath>
#include<map>
#include<memory>
#include<random>
#include<Eigen/Core>

//評価関数のパラメータをまとめたもの
//型は実際のパラメータ(int16_t)と、学習時のパラメータ,勾配(float)を想定
using DefaultEvalType = float;
using CalcType = float;
using LearnEvalType = float;

using Vec = Eigen::VectorXf;
using TeacherType = std::vector<CalcType>;
using Feature = std::vector<CalcType>;
const std::string DEFAULT_FILE_NAME = "model.bin";

constexpr int32_t POLICY_DIM = 64;

#ifdef USE_CATEGORICAL
constexpr int32_t BIN_SIZE = 51;
constexpr int32_t OUTPUT_DIM = POLICY_DIM + BIN_SIZE;
constexpr double VALUE_WIDTH = 1.0 / BIN_SIZE;
using ValueType = std::array<CalcType, BIN_SIZE>;
#else
constexpr int32_t OUTPUT_DIM = POLICY_DIM + 1;
using ValueType = CalcType;
#endif
constexpr int32_t INPUT_DIM = 64;
constexpr int32_t HIDDEN_DIM = 128;
constexpr int32_t LAYER_NUM = 2;

//LAYER_NUMを変えたらここの行列サイズも変えること
constexpr std::array<int32_t, 2> MATRIX_SIZE[LAYER_NUM] = {
    {HIDDEN_DIM, INPUT_DIM},
    {OUTPUT_DIM, HIDDEN_DIM}
};

//活性化関数の種類:オフにするとsigmoid
#define USE_ACTIVATION_RELU

template<typename T>
class EvalParams {
public:
    //初期化類
    EvalParams();
    void clear();
    void initRandom();

    //IO
    void readFile(std::string file_name = DEFAULT_FILE_NAME);
    void writeFile(std::string file_name = DEFAULT_FILE_NAME);

    //すべてのパラメータに同じ操作をする場合これを使う
    template<typename Function> void forEach(Function f);
    template<typename Function> void forEach(Function f) const;

    //統計:全てforEachで書ける
    double sumAbs() const;
    double maxAbs() const;

    //コピー
    void copy(const EvalParams<DefaultEvalType>& source);
    void roundCopy(const EvalParams<LearnEvalType>& source);

    using Mat = Eigen::Matrix<T, -1, -1>;

    std::array<Mat, LAYER_NUM> w;
    std::array<Mat, LAYER_NUM> b;
};

template<typename T>
inline EvalParams<T>::EvalParams() {
    clear();
}

template<typename T>
inline void EvalParams<T>::clear() {
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        w[i] = Mat::Zero(MATRIX_SIZE[i][0], MATRIX_SIZE[i][1]);
        b[i] = Mat::Zero(MATRIX_SIZE[i][0], 1);
    }
}

template<typename T>
inline void EvalParams<T>::initRandom() {
#ifdef USE_ACTIVATION_RELU
    //Heの初期化
    std::random_device seed;
    std::default_random_engine engine(seed());
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        int32_t fan_in = MATRIX_SIZE[i][1];
        std::normal_distribution<float> dist(0.0f, (float)std::sqrt(2.0 / fan_in));
        for (int32_t j = 0; j < MATRIX_SIZE[i][0]; j++) {
            for (int32_t k = 0; k < MATRIX_SIZE[i][1]; k++) {
                w[i](j, k) = dist(engine);
            }
            b[i](j) = dist(engine);
        }
    }
#else
    //Glorotの初期化
    std::random_device seed;
    std::default_random_engine engine(seed());
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        int32_t fan_in  = MATRIX_SIZE[i][1];
        int32_t fan_out = MATRIX_SIZE[i][0];
        std::normal_distribution<float> dist(0.0f, (float)std::sqrt(2.0 / (fan_in + fan_out)));
        for (int32_t j = 0; j < MATRIX_SIZE[i][0]; j++) {
            for (int32_t k = 0; k < MATRIX_SIZE[i][1]; k++) {
                w[i](j, k) = dist(engine);
            }
            b[i](j) = dist(engine);
        }
    }
#endif
}
template<typename T>
inline void EvalParams<T>::readFile(std::string file_name) {
    std::ifstream ifs(file_name, std::ios::binary);
    if (ifs.fail()) {
        std::cerr << file_name << " cannot open (mode r)" << std::endl;
        clear();
        return;
    }
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        ifs.read(reinterpret_cast<char*>(&w[i](0, 0)), w[i].size() * sizeof(T));
        ifs.read(reinterpret_cast<char*>(&b[i](0, 0)), b[i].size() * sizeof(T));
}
}

template<typename T>
inline void EvalParams<T>::writeFile(std::string file_name) {
    std::ofstream ofs(file_name, std::ios::binary | std::ios::trunc);
    if (ofs.fail()) {
        std::cerr << file_name << " cannot open (mode w)" << std::endl;
        return;
    }
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        ofs.write(reinterpret_cast<char*>(&w[i](0, 0)), w[i].size() * sizeof(T));
        ofs.write(reinterpret_cast<char*>(&b[i](0, 0)), b[i].size() * sizeof(T));
    }
}

template<typename T>
inline double EvalParams<T>::sumAbs() const {
    double sum = 0.0;
    forEach([&](const T value) { sum += std::abs(value); });
    return sum;
}

template<typename T>
inline double EvalParams<T>::maxAbs() const {
    double max_val = 0.0;
    forEach([&](const T value) { max_val = std::max(max_val, (double)std::abs(value)); });
    return max_val;
}

template<typename T>
template<typename Function>
void EvalParams<T>::forEach(Function f) {
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        for (int32_t j = 0; j < MATRIX_SIZE[i][0]; j++) {
            for (int32_t k = 0; k < MATRIX_SIZE[i][1]; k++) {
                f(w[i](j, k));
            }
            f(b[i](j, 0));
        }
    }
}

template<typename T>
template<typename Function>
void EvalParams<T>::forEach(Function f) const {
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        for (int32_t j = 0; j < MATRIX_SIZE[i][0]; j++) {
            for (int32_t k = 0; k < MATRIX_SIZE[i][1]; k++) {
                f(w[i](j, k));
            }
            f(b[i](j, 0));
        }
    }
}

template<typename T>
inline void EvalParams<T>::copy(const EvalParams<DefaultEvalType>& source) {
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        for (int32_t j = 0; j < MATRIX_SIZE[i][0]; j++) {
            for (int32_t k = 0; k < MATRIX_SIZE[i][1]; k++) {
                w[i](j, k) = source.w[i](j, k);
                b[i](j, k) = source.b[i](j, k);
            }
        }
    }
}

template<typename T>
inline void EvalParams<T>::roundCopy(const EvalParams<LearnEvalType>& source) {
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        for (int32_t j = 0; j < MATRIX_SIZE[i][0]; j++) {
            for (int32_t k = 0; k < MATRIX_SIZE[i][1]; k++) {
                w[i](j, k) = (T)std::round(source.w[i](j, k));
                b[i](j, k) = (T)std::round(source.b[i](j, k));
            }
        }
    }
}

extern std::unique_ptr<EvalParams<DefaultEvalType>> eval_params;

#endif // !EVAL_PARAMS_HPP