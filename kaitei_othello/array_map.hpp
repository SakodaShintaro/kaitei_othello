#pragma once

#ifndef ARRAYMAP_HPP
#define ARRAYMAP_HPP

#include<cstdint>
#include<cstring>

//Type型を返す配列であり、pairのinitializer_listで初期化できるもの
template<class Type, int32_t Size>
class ArrayMap {
public:
    constexpr ArrayMap() {
        // 速度低下を防止するため、特にゼロ初期化等は行わない
    }

    ArrayMap(std::initializer_list<Type> list) {
        assert(list.size() <= Size);
        std::copy(list.begin(), list.end(), begin());
    }

    ArrayMap(std::initializer_list<std::pair<int32_t, Type>> list) {
        for (const auto& pair : list) {
            array_[pair.first] = pair.second;
        }
    }

    Type* begin() {
        return &array_[0];
    }

    Type* end() {
        return begin() + Size;
    }

    constexpr const Type* begin() const {
        return &array_[0];
    }

    constexpr const Type* end() const {
        return begin() + Size;
    }

    Type& operator[](int32_t key) {
        return array_[key];
    }

    const Type& operator[](int32_t key) const {
        return array_[key];
    }

    constexpr size_t size() const {
        return Size;
    }

    void clear() {
        std::memset(&array_[0], 0, sizeof(Type) * Size);
    }
private:
    Type array_[Size];
};

#endif //!ARRAYMAP_HPP
