#include"position.hpp"
#include"piece.hpp"
#include"move.hpp"
#include"common.hpp"
#include"usi_options.hpp"
#include<iostream>
#include<cstdio>
#include<ctime>
#include<bitset>
#include<cassert>
#include<iterator>
#include<algorithm>
#include<set>

int64_t Position::HashSeed[PieceNum][SquareNum];

Position::Position(const EvalParams<DefaultEvalType>& eval_params) : eval_params_(eval_params) {
    init();
}

void Position::init() {
    //盤上の初期化
    for (int32_t i = 0; i < SquareNum; i++) {
        board_[i] = WALL;
    }
    for (Square sq : SquareList) {
        board_[sq] = EMPTY;
    }

    //後手の駒
    board_[SQ54] = board_[SQ45] = WHITE_PIECE;

    //先手の駒
    board_[SQ44] = board_[SQ55] = BLACK_PIECE;

    //手番
    color_ = BLACK;

    //手数
    turn_number_ = 0;

    //ハッシュ値の初期化
    initHashValue();

    //局面の評価値
    initPolicyAndValue();

    kifu_.clear();
    kifu_.reserve(512);

    //Bitboard
    occupied_bb_[BLACK] = Bitboard(0);
    occupied_bb_[WHITE] = Bitboard(0);
    for (Square sq : SquareList) {
        if (board_[sq] != EMPTY) {
            occupied_bb_[board_[sq]] |= SQUARE_BB[sq];
        }
    }
    occupied_all_ = occupied_bb_[BLACK] | occupied_bb_[WHITE];
}

void Position::print() const {
    //盤上
    std::printf("ABCDEFGH\n");
    std::printf("--------\n");
    for (int r = Rank1; r <= Rank8; r++) {
        for (int f = File8; f >= File1; f--) {
            std::cout << PieceToSfenStr[board_[FRToSquare[f][r]]];
        }
        printf("|%d\n", r);
    }

    //手番
    printf("手番:");
    if (color_ == BLACK) printf("先手\n");
    else printf("後手\n");

    //手数
    printf("手数:%d\n", turn_number_);

    //最後の手
    if (!kifu_.empty()) {
        printf("最後の手:");
        lastMove().printWithScore();
    }

    printAllMoves();

    //評価値
    auto output = makeOutput();
#ifdef USE_CATEGORICAL
    std::vector<CalcType> categorical_distribution(BIN_SIZE);
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        categorical_distribution[i] = output(POLICY_DIM + i);
    }
    categorical_distribution = softmax(categorical_distribution);

    CalcType value = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        printf("p[%f] = %f\n", VALUE_WIDTH * (0.5 + i), categorical_distribution[i]);
        value += (CalcType)(VALUE_WIDTH * (0.5 + i) * categorical_distribution[i]);
    }
    printf("value = %f\n", value);
#else
    printf("value = %f\n", standardSigmoid(output(POLICY_DIM)));
#endif

    printf("ハッシュ値:%lld\n", hash_value_);
}

void Position::printAllMoves() const {
    std::vector<Move> moves = generateAllMoves();
    printf("全合法手の数:%zu\n", moves.size());
    for (Move move : moves) move.print();
}

void Position::printHistory() const {
    printf("print history\n");
    for (Move move : kifu_) move.print();
    printf("\n");
}

void Position::printForDebug() const {
    print();
    printHistory();
}

void Position::doMove(const Move move) {
#if DEBUG
    if (!isLegalMove(move)) {
        printForDebug();
        std::cout << "違法だった手:";
        move.print();
        isLegalMove(move);
        undo();
        printAllMoves();
        assert(false);
    }
#endif

    if (move == NULL_MOVE) {
        doNullMove();
        return;
    }

    //動かす前の状態を残しておく
    std::vector<Piece> board(SquareNum);
    for (int32_t i = 0; i < SquareNum; i++) {
        board[i] = board_[i];
    }
    stack_.emplace_back(board);

    //実際に動かす
    //8方向を一つずつ見ていって反転できる駒があったら反転する
    Piece p = board_[move.to()] = (Piece)color_;

    hash_values_.push_back(hash_value_);

    //ハッシュ値を変更
    hash_value_ ^= HashSeed[p][move.to()];

    for (Dir d : DirList) {
        bool is_there_enemy = false;
        bool isOK = false;
        for (int32_t sq = move.to() + d; board_[sq] != WALL; sq += d) {
            if (board_[sq] == oppositeColor(p)) {
                is_there_enemy = true;
            } else if (board_[sq] == p) {
                if (is_there_enemy) {
                    isOK = true;
                }
                break;
            } else {
                break;
            }
        }
        if (!isOK) {
            continue;
        }

        //実際に駒を変更する
        for (int32_t sq = move.to() + d; board_[sq] != p; sq += d) {
            board_[sq] = p;
            occupied_bb_[p] |= SQUARE_BB[sq];
            occupied_bb_[oppositeColor(p)] ^= SQUARE_BB[sq];
            hash_value_ ^= HashSeed[oppositeColor(p)][sq];
            hash_value_ ^= HashSeed[p][sq];
        }
    }

    //occupied_all_を更新
    occupied_all_ = occupied_bb_[BLACK] | occupied_bb_[WHITE];

    //手番の更新
    color_ = ~color_;

    //手数の更新
    turn_number_++;

    //棋譜に指し手を追加
    kifu_.push_back(move);

    //1bit目を0にする
    hash_value_ ^= 1;

    already_calc_ = false;
}

void Position::undo() {
    const Move last_move = kifu_.back();
    kifu_.pop_back();

    //手番を戻す(このタイミングでいいかな?)
    color_ = ~color_;

    //盤の状態はstack_から戻して
    for (int32_t i = 0; i < SquareNum; i++) {
        board_[i] = stack_.back()[i];
    }
    stack_.pop_back();

    //occupied_all_を更新
    occupied_all_ = occupied_bb_[BLACK] | occupied_bb_[WHITE];

    //ハッシュの更新
    hash_value_ = hash_values_.back();
    hash_values_.pop_back();

    //手数
    turn_number_--;
 
    already_calc_ = false;
}

void Position::doNullMove() {
    //スタックの変更
    std::vector<Piece> board(SquareNum);
    for (int32_t i = 0; i < SquareNum; i++) {
        board[i] = board_[i];
    }
    stack_.emplace_back(board);

    hash_values_.push_back(hash_value_);

    //手番の更新
    color_ = ~color_;

    //手数の更新
    turn_number_++;

    //hashの手番要素を更新
    hash_value_ ^= 1;

    //NULL_MOVEを棋譜に追加
    kifu_.push_back(NULL_MOVE);

    //反転するから評価はしていない
    already_calc_ = false;
}

void Position::undoNullMove() {
    kifu_.pop_back();

    //手番を戻す(このタイミングでいいかな?)
    color_ = ~color_;

    //ハッシュの更新(手番分)
    hash_value_ ^= 1;

    //手数
    turn_number_--;
}

bool Position::isLegalMove(const Move move) const {
    //間に敵駒がありその先に自駒があれば良い
    if (board_[move.to()] != EMPTY) {
        return false;
    }
    Piece p = (Piece)color_;
    for (Dir d : DirList) {
        bool ok = false;
        for (int32_t sq = move.to() + d; board_[sq] != WALL; sq += d) {
            if (board_[sq] == oppositeColor(p)) {
                ok = true;
            } else if (board_[sq] == p) {
                if (ok) {
                    return true;
                }
                break;
            } else {
                break;
            }
        }
    }
    return false;
}

bool Position::isFinish() const {
    //パスが2回続いたら終了
    int32_t N = (int32_t)kifu_.size();
    return (N >= 2 && kifu_[N - 1] == NULL_MOVE && kifu_[N - 2] == NULL_MOVE);
}

void Position::initHashSeed() {
    std::mt19937_64 rnd(5981793);
    for (int32_t piece = BLACK_PIECE; piece <= WHITE_PIECE; piece++) {
        for (Square sq : SquareList) {
            HashSeed[piece][sq] = rnd();
        }
    }
}

std::vector<Move> Position::generateAllMoves() const {
    std::vector<Move> moves;
    for (Square sq : SquareList) {
        Move move(sq, color_);
        if (isLegalMove(move)) {
            moves.push_back(move);
        }
    }
    if (moves.size() == 0) {
        moves.push_back(NULL_MOVE);
    }
    return moves;
}

std::vector<Move> Position::scoredAndSortedMoves() {
    auto moves = generateAllMoves();
    if (!already_calc_) {
        initPolicyAndValue();
    }
    for (Move& move : moves) {
        move.score = output_(move.toLabel());
    }
    sort(moves.begin(), moves.end(), std::greater<Move>());
    return moves;
}

void Position::initHashValue() {
    hash_value_ = 0;
    for (auto sq : SquareList) {
        hash_value_ ^= HashSeed[board_[sq]][sq];
    }
    hash_value_ &= ~1; //これで1bit目が0になる(先手番を表す)
}

void Position::loadData(std::array<int64_t, 3> bb) {
    //盤上の初期化
    for (int32_t i = 0; i < SquareNum; i++) {
        board_[i] = WALL;
    }
    for (Square sq : SquareList) {
        board_[sq] = EMPTY;
    }

    //bbに従って駒を配置する
    for (int32_t c : { BLACK, WHITE }) {
        for (Square sq : SquareList) {
            if (bb[c] & SQUARE_BB[sq]) {
                board_[sq] = Piece(c);
            }
        }
        occupied_bb_[c] = bb[c];
    }

    //手番
    color_ = Color(bb[2]);

    //手数
    turn_number_ = 0;

    //ハッシュ値の初期化
    initHashValue();

    //局面の評価値
    initPolicyAndValue();

    kifu_.clear();
    kifu_.reserve(512);

    //Bitboard
    occupied_all_ = occupied_bb_[BLACK] | occupied_bb_[WHITE];
}