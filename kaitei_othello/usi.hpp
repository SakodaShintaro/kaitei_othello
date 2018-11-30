#ifndef USI_HPP
#define USI_HPP

#include"common.hpp"
#include"position.hpp"
#include"searcher.hpp"
#include"move.hpp"
#include<string>

class NBoardProtocol {
public:
    NBoardProtocol() : root_(*eval_params) {}
    void loop();
    void vsHuman();
    void vsAI();
private:
    Position root_;
};

#endif