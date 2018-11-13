#ifndef USI_HPP
#define USI_HPP

#include"common.hpp"
#include"position.hpp"
#include"searcher.hpp"
#include"move.hpp"
#include"thread.hpp"
#include<string>

class NBoardProtocol {
public:
    void loop();
    void vsHuman();
};

#endif
