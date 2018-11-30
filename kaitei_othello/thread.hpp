#pragma once

#ifndef THREAD_HPP
#define THREAD_HPP

#include<vector>
#include<thread>
#include<atomic>
#include<iostream>
#include<mutex>
#include<condition_variable>
#include"types.hpp"
#include"move.hpp"
#include"searcher.hpp"

class Position;

class Thread {
protected:
    std::thread thread_;
    uint32_t id_;
    bool exit_, searching_;
    std::mutex mutex_;
    std::condition_variable sleep_condition_;
    AlphaBetaSearcher searcher_;
public:
    Thread(uint32_t id) : id_(id), exit_(false), searching_(false), searcher_(AlphaBetaSearcher::Role::SLAVE) {
        thread_ = std::thread(&Thread::idleLoop, this);
    }
    ~Thread() {
        mutex_.lock();
        exit_ = true;
        sleep_condition_.notify_one();
        mutex_.unlock();
        thread_.join();
    }

    void idleLoop();

    virtual void startSearch() {
        std::unique_lock<std::mutex> lock(mutex_);
        searching_ = true;
        sleep_condition_.notify_one();
    }

    void waitForFinishSearch() {
        std::unique_lock<std::mutex> lock(mutex_);
        sleep_condition_.wait(lock, [this]() {
            return !searching_;
        });
    }
};

struct ThreadPool : public std::vector<std::unique_ptr<Thread>> {
    void init();
};
extern ThreadPool threads;

class MainThread : public Thread {
private:

public:
    MainThread(uint32_t id) : Thread(id) {
        searcher_.setRole(AlphaBetaSearcher::Role::MAIN);
    }

    virtual void startSearch() {
        std::unique_lock<std::mutex> lock(mutex_);
        searching_ = true;
        sleep_condition_.notify_one();
        for (int i = 1; i < threads.size(); ++i) {
            threads[i]->startSearch();
        }
    }
};

#endif // !THREAD_HPP