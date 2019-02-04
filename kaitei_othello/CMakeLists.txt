cmake_minimum_required(VERSION 3.10)
project(kaitei_othello)

#共通する設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories("../../eigen")

#使用するソースも共通なので変数でまとめる
set(SRCS alphabeta_searcher.cpp
         alphazero_trainer.cpp
         base_trainer.cpp
         bitboard.cpp
         calcScore.cpp
         game.cpp
         hash_table.cpp
         main.cpp
         MCTSearcher.cpp
         network.cpp
         piece.cpp
         position.cpp
         replay_buffer.cpp
         square.cpp
         test.cpp
         uct_hash_table.cpp
         usi.cpp
         )

add_executable(Scalar ${SRCS})
add_executable(Categorical ${SRCS})
target_compile_definitions(Categorical PUBLIC USE_CATEGORICAL)
target_link_libraries(Scalar pthread stdc++fs)
target_link_libraries(Categorical pthred stdc++fs)