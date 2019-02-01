#include"network.hpp"
#include"common.hpp"
#include"move.hpp"
#include<fstream>
#include<iostream>

Vec Network::activationFunction(const Vec & x) {
#ifdef USE_ACTIVATION_RELU
    return relu(x);
#else
    return sigmoid(x);
#endif
}

Vec Network::d_activationFunction(const Vec & x) {
#ifdef USE_ACTIVATION_RELU
    return d_relu(x);
#else
    return d_sigmoid(x);
#endif
}

#ifdef USE_ACTIVATION_RELU

Vec Network::relu(const Vec& x) {
    Vec result = x;
    for (int i = 0; i < result.size(); i++) {
        result(i) = (result(i) >= 0.0f ? result(i) : 0.0f);
    }
    return result;
}

Vec Network::d_relu(const Vec& x) {
    Vec result = x;
    for (int i = 0; i < result.size(); i++) {
        result(i) = (result(i) >= 0.0f ? 1.0f : 0.0f);
    }
    return result;
}

#else

Vec Network::sigmoid(const Vec & x) {
    Vec result = x;
    for (int i = 0; i < result.size(); i++) {
        result(i) = (CalcType)standardSigmoid(result(i));
    }
    return result;
}

Vec Network::d_sigmoid(const Vec & x) {
    Vec result = x;
    for (int i = 0; i < result.size(); i++) {
        result(i) = (CalcType)d_standardSigmoid(result(i));
    }
    return result;
}

#endif