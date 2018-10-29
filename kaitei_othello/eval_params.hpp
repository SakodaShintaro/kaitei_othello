#pragma once
#ifndef EVAL_PARAMS_HPP
#define EVAL_PARAMS_HPP

#include"piece.hpp"
#include"eval_elements.hpp"
#include"square.hpp"
#include<climits>
#include<cmath>
#include<map>
#include<memory>
#include<random>
#ifdef USE_NN
#include<Eigen/Core>
#endif

constexpr uint64_t SqNum = 81;

//評価関数のパラメータをまとめたもの
//型は実際のパラメータ(int16_t)と、学習時のパラメータ,勾配(float)を想定
#ifdef USE_NN
using DefaultEvalType = float;
using CalcType = float;
using Vec = Eigen::VectorXf;
using TeacherType = std::vector<CalcType>;
const std::string DEFAULT_FILE_NAME = "model.bin";

//#define SMALL_OUTPUT
#ifdef SMALL_OUTPUT
constexpr int32_t POLICY_DIM = 81 * 10; // = pieceToIndex[BLACK_KING]
#else
constexpr int32_t POLICY_DIM = 64;
#endif

#ifdef USE_CATEGORICAL
constexpr int32_t BIN_SIZE = 11;
constexpr int32_t OUTPUT_DIM = POLICY_DIM + BIN_SIZE;
constexpr double VALUE_WIDTH = 1.0 / BIN_SIZE;
#else
constexpr int32_t OUTPUT_DIM = POLICY_DIM + 1;
#endif
constexpr int32_t INPUT_DIM = 65; //64(盤面) + 1(手番)
constexpr int32_t HIDDEN_DIM = 128;
constexpr int32_t LAYER_NUM = 2;

//LAYER_NUMを変えたらここの行列サイズも変えること
constexpr std::array<int32_t, 2> MATRIX_SIZE[LAYER_NUM] = {
    {HIDDEN_DIM, INPUT_DIM},
    {OUTPUT_DIM, HIDDEN_DIM}
};

//活性化関数の種類:オフにするとsigmoid
//#define USE_ACTIVATION_RELU

#else
using DefaultEvalType = int16_t;
using TeacherType = double;
const std::string DEFAULT_FILE_NAME = "parameters.bin";
#endif
using LearnEvalType = float;

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

    //統計:全てforEachで書ける(=KPPTとNNで変わらない)
    double sumAbs() const;
    double maxAbs() const;
#ifdef USE_NN
    void printHistgram(int64_t bin_size = 1) const;
    void printBias() const;
#else
    void printHistgram(int64_t bin_size = 100) const;
#endif

    //コピー
    void copy(const EvalParams<DefaultEvalType>& source);
    void roundCopy(const EvalParams<LearnEvalType>& source);

#ifdef USE_NN
    using Mat = Eigen::Matrix<T, -1, -1>;

    std::array<Mat, LAYER_NUM> w;
    std::array<Mat, LAYER_NUM> b;
#else
    std::array<T, ColorNum> kkp[SqNum][SqNum][PieceStateNum];
    std::array<T, ColorNum> kpp[SqNum][PieceStateNum][PieceStateNum];
#endif
};

template<typename T>
inline EvalParams<T>::EvalParams() {
    clear();
}

template<typename T>
inline void EvalParams<T>::clear() {
#ifdef USE_NN
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        w[i] = Mat::Zero(MATRIX_SIZE[i][0], MATRIX_SIZE[i][1]);
        b[i] = Mat::Zero(MATRIX_SIZE[i][0], 1);
    }
#else
    forEach([&](T& value) { value = 0; });
#endif
}

template<typename T>
inline void EvalParams<T>::initRandom() {
#ifdef USE_NN
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
#else
    int i = 0;
    forEach([&](T& value) { value = i++; });
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
#ifdef USE_NN
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        ifs.read(reinterpret_cast<char*>(&w[i](0, 0)), w[i].size() * sizeof(T));
        ifs.read(reinterpret_cast<char*>(&b[i](0, 0)), b[i].size() * sizeof(T));
}
#else
    ifs.read(reinterpret_cast<char*>(&kkp[0][0][0][0]), SqNum * SqNum * PieceStateNum * ColorNum *  sizeof(T));
    ifs.read(reinterpret_cast<char*>(&kpp[0][0][0][0]), SqNum * PieceStateNum * PieceStateNum * ColorNum * sizeof(T));
#endif
}

template<typename T>
inline void EvalParams<T>::writeFile(std::string file_name) {
    std::ofstream ofs(file_name, std::ios::binary | std::ios::trunc);
    if (ofs.fail()) {
        std::cerr << file_name << " cannot open (mode w)" << std::endl;
        return;
    }
#ifdef USE_NN
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        ofs.write(reinterpret_cast<char*>(&w[i](0, 0)), w[i].size() * sizeof(T));
        ofs.write(reinterpret_cast<char*>(&b[i](0, 0)), b[i].size() * sizeof(T));
    }
#else
    ofs.write(reinterpret_cast<char*>(&kkp[0][0][0][0]), SqNum * SqNum * PieceStateNum * ColorNum * sizeof(T));
    ofs.write(reinterpret_cast<char*>(&kpp[0][0][0][0]), SqNum * PieceStateNum * PieceStateNum * ColorNum * sizeof(T));
#endif
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
inline void EvalParams<T>::printHistgram(int64_t bin_size) const {
    std::cout << "sumAbs() = " << sumAbs() << std::endl;
    std::map<int64_t, uint64_t> histgram;
    forEach([&](T value) {
        if (value == 0) {
            //0はそのまま
            histgram[static_cast<int64_t>(value)]++;
        } else if (value > 0) {
            //正の数はたとえばbin_size = 5のとき1~5が1, 6~10が2, ...となるように計算する 
            histgram[static_cast<int64_t>((value + bin_size - 1) / bin_size)]++;
        } else {
            //負の数も-1~-5が-1, -6~-10が-2, ...となるように計算する
            histgram[static_cast<int64_t>((value - bin_size + 1) / bin_size)]++;
        }
    });
#ifdef USE_NN
    double all_num = 0.0;
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        all_num += MATRIX_SIZE[i][0] * (MATRIX_SIZE[i][1] + 1);
    }
#else
    double all_num = (SqNum * SqNum * PieceStateNum + SqNum * PieceStateNum * PieceStateNum) * ColorNum;
#endif
    for (auto e : histgram) {
        if (e.first == 0) {
            //0はそのまま数えた
            printf("      0       : %9llu (%6.2f)\n", e.second, (double)e.second / all_num * 100.0);
        } else if (e.first > 0) {
            printf("%6lld~%6lld : %9llu (%6.2f)\n", (e.first - 1) * bin_size + 1, e.first * bin_size, e.second, (double)e.second / all_num * 100.0);
        } else {
            printf("%6lld~%6lld : %9llu (%6.2f)\n", (e.first + 1) * bin_size - 1, e.first * bin_size, e.second, (double)e.second / all_num * 100.0);
        }
    }
    std::cout << std::endl;
}

template<typename T>
template<typename Function>
void EvalParams<T>::forEach(Function f) {
#ifdef USE_NN
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        for (int32_t j = 0; j < MATRIX_SIZE[i][0]; j++) {
            for (int32_t k = 0; k < MATRIX_SIZE[i][1]; k++) {
                f(w[i](j, k));
            }
            f(b[i](j, 0));
        }
    }
#else
    for (int k1 = 0; k1 < SqNum; k1++) {
        for (int p1 = 0; p1 < PieceStateNum; p1++) {
            for (int t = 0; t < ColorNum; t++) {
                for (int k2 = 0; k2 < SqNum; k2++) {
                    f(kkp[k1][k2][p1][t]);
                }
                for (int p2 = 0; p2 < PieceStateNum; p2++) {
                    f(kpp[k1][p1][p2][t]);
                }
            }
        }
    }
#endif
}

template<typename T>
template<typename Function>
void EvalParams<T>::forEach(Function f) const {
#ifdef USE_NN
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        for (int32_t j = 0; j < MATRIX_SIZE[i][0]; j++) {
            for (int32_t k = 0; k < MATRIX_SIZE[i][1]; k++) {
                f(w[i](j, k));
            }
            f(b[i](j, 0));
        }
    }
#else
    for (int k1 = 0; k1 < SqNum; k1++) {
        for (int p1 = 0; p1 < PieceStateNum; p1++) {
            for (int t = 0; t < ColorNum; t++) {
                for (int k2 = 0; k2 < SqNum; k2++) {
                    f(kkp[k1][k2][p1][t]);
                }
                for (int p2 = 0; p2 < PieceStateNum; p2++) {
                    f(kpp[k1][p1][p2][t]);
                }
            }
        }
    }
#endif
}

template<typename T>
inline void EvalParams<T>::copy(const EvalParams<DefaultEvalType>& source) {
#ifdef USE_NN
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        for (int32_t j = 0; j < MATRIX_SIZE[i][0]; j++) {
            for (int32_t k = 0; k < MATRIX_SIZE[i][1]; k++) {
                w[i](j, k) = source.w[i](j, k);
                b[i](j, k) = source.b[i](j, k);
            }
        }
    }
#else
    for (int k1 = 0; k1 < SqNum; ++k1) {
        for (int p1 = 0; p1 < PieceStateNum; ++p1) {
            for (int k2 = 0; k2 < SqNum; ++k2) {
                for (int t = 0; t < ColorNum; ++t) {
                    kkp[k1][k2][p1][t] = static_cast<T>(source.kkp[k1][k2][p1][t]);
                }
            }
            for (int p2 = 0; p2 < PieceStateNum; ++p2) {
                for (int t = 0; t < ColorNum; ++t) {
                    kpp[k1][p1][p2][t] = static_cast<T>(source.kpp[k1][p1][p2][t]);
                }
            }
        }
    }
#endif
}

template<typename T>
inline void EvalParams<T>::roundCopy(const EvalParams<LearnEvalType>& source) {
#ifdef USE_NN
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        for (int32_t j = 0; j < MATRIX_SIZE[i][0]; j++) {
            for (int32_t k = 0; k < MATRIX_SIZE[i][1]; k++) {
                w[i](j, k) = (T)std::round(source.w[i](j, k));
                b[i](j, k) = (T)std::round(source.b[i](j, k));
            }
        }
    }
#else
    for (int k1 = 0; k1 < SqNum; ++k1) {
        for (int p1 = 0; p1 < PieceStateNum; ++p1) {
            for (int k2 = 0; k2 < SqNum; ++k2) {
                for (int t = 0; t < ColorNum; ++t) {
                    kkp[k1][k2][p1][t] = static_cast<T>(std::round(source.kkp[k1][k2][p1][t]));
                }
            }
            for (int p2 = 0; p2 < PieceStateNum; ++p2) {
                for (int t = 0; t < ColorNum; ++t) {
                    kpp[k1][p1][p2][t] = static_cast<T>(std::round(source.kpp[k1][p1][p2][t]));
                }
            }
        }
    }
#endif
}

#ifdef USE_NN
template<typename T>
inline void EvalParams<T>::printBias() const {
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        std::cout << i + 1 << "層目" << std::endl;
        std::cout << b[i] << std::endl;
    }
}
#endif

extern std::unique_ptr<EvalParams<DefaultEvalType>> eval_params;

#endif // !EVAL_PARAMS_HPP
