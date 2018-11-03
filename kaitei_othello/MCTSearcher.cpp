#include"MCTSearcher.hpp"
#include"shared_data.hpp"
#include"network.hpp"
#include"usi_options.hpp"
#include"operate_params.hpp"

#ifdef USE_MCTS

void MCTSearcher::think() {
    //思考開始時間をセット
    start_ = std::chrono::steady_clock::now();

    //コピーして使う
    Position root = shared_data.root;

    //思考する局面の表示
    root.print();

    //古いハッシュを削除
    hash_table_.deleteOldHash(root);

    //ルートノードの展開
    current_root_index_ = expandNode(root);
    auto& current_node = hash_table_[current_root_index_];
    auto child_num = current_node.child_num;
    auto root_moves = current_node.legal_moves;

    //合法手が0だったら投了
    if (child_num == 0) {
        std::cout << "bestmove resign" << std::endl;
        return;
    }

    //Alpha Zeroの論文と同じディリクレノイズ
    constexpr double epsilon = 0.25;
    auto dirichlet = dirichletDistribution(current_node.child_num, 0.015);
    for (int32_t i = 0; i < current_node.child_num; i++) {
        current_node.nn_rates[i] = (CalcType)((1.0 - epsilon) * current_node.nn_rates[i] + epsilon * dirichlet[i]);
    }

    //合法手が1つだったらすぐ送る
    //これもUSIオプション化した方が良いか
    //if (child_num == 1) {
    //    std::cout << "bestmove " << root_moves[0] << std::endl;
    //    return;
    //}

    //指定された手数までソフトマックスランダムにより指し手を決定
    if (root.turn_number() < usi_option.random_turn) {
        auto prob = softmax(current_node.nn_rates, (CalcType)usi_option.temperature);
        int32_t index = randomChoise(prob);
        Move random_move = current_node.legal_moves[index];
        std::cout << "bestmove " << random_move << std::endl;
        return;
    }

    //初期化
    playout_num = 0;

    //プレイアウトを繰り返す
    //探索回数が閾値を超える、または探索が打ち切られたらループを抜ける
    //最大訪問手と最高勝率手が一致しないとき延長するというテクニックがあるらしい(未実装)
    while (playout_num < usi_option.playout_limit) {
        //探索回数を1回増やす
        playout_num++;

        //1回プレイアウト
        uctSearch(root, current_root_index_);

        //探索を打ち切るか確認
        if (shouldStop() || !hash_table_.hasEnoughSize()) {
            break;
        }

        if (playout_num != 0 && playout_num % 30000 == 0) {
            printUSIInfo();
        }
    }

    const auto& child_move_counts = current_node.child_move_counts;

    //for debug
#ifdef USE_CATEGORICAL
    for (int32_t i = 0; i < child_num; i++) {
        printf("%3d: move_count = %6d, nn_rate = %.5f, ", i, child_move_counts[i], current_node.nn_rates[i]);
        for (int32_t j = 0; j < BIN_SIZE; j++) {
            printf("%.2f ", (child_move_counts[i] > 0 ? current_node.child_wins[i][j] / child_move_counts[i] : 0));
        }
        root_moves[i].print();
    }
#else
    for (int32_t i = 0; i < child_num; i++) {
        printf("%3d: move_count = %6d, nn_rate = %.5f, win_rate = %7.5f, ", i, child_move_counts[i],
            current_node.nn_rates[i], (child_move_counts[i] > 0 ? current_node.child_wins[i] / child_move_counts[i] : 0));
        root_moves[i].print();
    }
#endif

    printUSIInfo();

    //訪問回数最大の手を選択
    int32_t best_index = (int32_t)(std::max_element(child_move_counts.begin(), child_move_counts.end()) - child_move_counts.begin());

    //選択した着手の勝率の算出
#ifdef USE_CATEGORICAL
    double best_wp = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        double v = current_node.child_wins[best_index][i] / child_move_counts[best_index];
        best_wp += VALUE_WIDTH * (0.5 + i) * v;
    }
#else
    auto best_wp = (child_move_counts[best_index] == 0 ? 0.0 
        : current_node.child_wins[best_index] / child_move_counts[best_index]);
#endif
    assert(0.0 <= best_wp && best_wp <= 1.0);

    //閾値未満の場合は投了
    if (best_wp < sigmoid(usi_option.resign_score, CP_GAIN)) {
        std::cout << "bestmove resign" << std::endl;
    } else {
        std::cout << "bestmove " << root_moves[best_index] << std::endl;
    }
}

std::pair<Move, TeacherType> MCTSearcher::thinkForGenerateLearnData(Position& root, int32_t playout_limit) {
    //思考開始時間をセット
    start_ = std::chrono::steady_clock::now();

    //古いハッシュを削除
    hash_table_.deleteOldHash(root);

    //ルートノードの展開
    current_root_index_ = expandNode(root);
    auto& current_node = hash_table_[current_root_index_];

    //合法手が0だったら投了
    if (current_node.child_num == 0) {
        return { NULL_MOVE, TeacherType() };
    }

    //NULL_MOVEだけでもすぐ返す
    if (current_node.child_num == 1 && current_node.legal_moves[0] == NULL_MOVE) {
        return { NULL_MOVE, TeacherType() };
    }

    //ノイズを加える
    //Alpha Zeroの論文と同じディリクレノイズ
    constexpr double epsilon = 0.25;
    auto dirichlet = dirichletDistribution(current_node.child_num, 0.015);
    for (int32_t i = 0; i < current_node.child_num; i++) {
        current_node.nn_rates[i] = (CalcType)((1.0 - epsilon) * current_node.nn_rates[i] + epsilon * dirichlet[i]);
    }

    //初期化
    playout_num = 0;

    //プレイアウトを繰り返す
    //探索回数が閾値を超える、または探索が打ち切られたらループを抜ける
    while (playout_num < (uint32_t)playout_limit) {
        //探索回数を1回増やす
        playout_num++;

        //1回プレイアウト
        uctSearch(root, current_root_index_);

        //探索を打ち切るか確認
        if (shouldStop() || !hash_table_.hasEnoughSize()) {
            break;
        }
    }

    const auto& child_move_counts = current_node.child_move_counts;

    //for debug
    //root.print();
    //auto child_num = current_node.child_num;
    //auto root_moves = current_node.legal_moves;
    //for (int32_t i = 0; i < child_num; i++) {
    //    printf("%3d: move_count = %6d, nn_rate = %.5f, win_rate = %7.5f, ", i, child_move_counts[i],
    //        current_node.nn_rates[i], (child_move_counts[i] > 0 ? current_node.child_wins[i] / child_move_counts[i] : 0));
    //    root_moves[i].print();
    //}
    
    // 訪問回数最大の手を選択する
    int32_t best_index = (int32_t)(std::max_element(child_move_counts.begin(), child_move_counts.end()) - child_move_counts.begin());

    //選択した着手の勝率の算出
#ifdef USE_CATEGORICAL
    double best_wp = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        double v = current_node.child_wins[best_index][i] / child_move_counts[best_index];
        best_wp += VALUE_WIDTH * (0.5 + i) * v;
    }
#else
    auto best_wp = (child_move_counts[best_index] == 0 ? 0.0
        : current_node.child_wins[best_index] / child_move_counts[best_index]);
#endif
    assert(0.0 <= best_wp && best_wp <= 1.0);

    if (best_wp < sigmoid(usi_option.resign_score, CP_GAIN)) {
        //閾値未満の場合は投了
        return { NULL_MOVE, TeacherType() };
    } else {
        //投了しない場合教師データを作成
        TeacherType teacher(OUTPUT_DIM, 0.0);

        //valueのセット
#ifdef USE_CATEGORICAL
        for (int32_t i = 0; i < BIN_SIZE; i++) {
            teacher[POLICY_DIM + i] = current_node.child_wins[best_index][i] / child_move_counts[best_index];
        }
#else
        teacher[POLICY_DIM] = (CalcType)best_wp;
#endif

        //最善手
        Move best_move = current_node.legal_moves[best_index];

        if (root.turn_number() < usi_option.random_turn) {
            //ランダムなら訪問回数に基づいた分布を得る
            std::vector<CalcType> distribution(current_node.child_num);
            for (int32_t i = 0; i < current_node.child_num; i++) {
                distribution[i] = (CalcType)child_move_counts[i] / current_node.move_count;
                assert(0.0 <= distribution[i] && distribution[i] <= 1.0);

                //分布を教師データにセット
                teacher[current_node.legal_moves[i].toLabel()] = distribution[i];
            }
            //分布に基づいて指し手を選択
            best_move = current_node.legal_moves[randomChoise(distribution)];
        } else {
            //訪問回数最大のもの = best_moveを選ぶような分布
            teacher[best_move.toLabel()] = 1.0;
        }
        
        best_move.score = (Score)best_wp;

        return { best_move, teacher };
    }
}

#ifdef USE_CATEGORICAL
std::array<CalcType, BIN_SIZE> MCTSearcher::uctSearch(Position & pos, Index current_index) {
#else
CalcType MCTSearcher::uctSearch(Position & pos, Index current_index) {
#endif
    auto& current_node = hash_table_[current_index];

    if (current_node.child_num == 0) {
#ifdef USE_CATEGORICAL
        std::array<CalcType, BIN_SIZE> lose_value_dist;
        for (int32_t i = 0; i < BIN_SIZE; i++) {
            lose_value_dist[i] = (i == 0 ? 1.0f : 0.0f);
        }
        return lose_value_dist;
#else
        return 0.0;
#endif
    }

    auto& child_indices = current_node.child_indices;

    // UCB値が最大の手を求める
#ifdef USE_CATEGORICAL
    const auto& child_move_counts = current_node.child_move_counts;
    int32_t selected_index = (int32_t)(std::max_element(child_move_counts.begin(), child_move_counts.end()) - child_move_counts.begin());
    double best_wp = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        double v = (child_move_counts[selected_index] == 0 ? 0.0 :
            current_node.child_wins[selected_index][i] / child_move_counts[selected_index]);
        best_wp += VALUE_WIDTH * (0.5 + i) * v;
    }
    auto next_index = selectMaxUcbChild(current_node, best_wp);
#else
    auto next_index = selectMaxUcbChild(current_node);
#endif

    // 選んだ手を着手
    pos.doMove(current_node.legal_moves[next_index]);

#ifdef USE_CATEGORICAL
    std::array<CalcType, BIN_SIZE> result;
#else
    CalcType result;
#endif
    // ノードの展開の確認
    if (pos.isFinish()) {
        //終了
#ifdef USE_CATEGORICAL
        result = reverseDist(onehotDist(pos.resultForTurn()));
#else
        result = (CalcType)(1.0 - pos.resultForTurn());
#endif
    } else if (child_indices[next_index] == UctHashTable::NOT_EXPANDED) {
        // ノードの展開
        auto index = expandNode(pos);
        child_indices[next_index] = index;
#ifdef USE_CATEGORICAL
        result = reverseDist(hash_table_[index].value_dist);
#else
        result = 1.0f - hash_table_[index].value_win;
#endif
    } else {
        // 手番を入れ替えて1手深く読む
#ifdef USE_CATEGORICAL
        auto retval = uctSearch(pos, child_indices[next_index]);
        result = reverseDist(retval);
#else
        result = 1.0f - uctSearch(pos, child_indices[next_index]);
#endif
    }

    // 探索結果の反映
    current_node.win_sum += result;
    current_node.move_count++;
    current_node.child_wins[next_index] += result;
    current_node.child_move_counts[next_index]++;

    // 手を戻す
    pos.undo();

    return result;
}

Index MCTSearcher::expandNode(Position& pos) {
    auto index = hash_table_.findSameHashIndex(pos.hash_value(), pos.turn_number());

    // 合流先が検知できればそれを返す
    if (index != hash_table_.size()) {
        return index;
    }

    // 空のインデックスを探す
    index = hash_table_.searchEmptyIndex(pos.hash_value(), pos.turn_number());

    auto& current_node = hash_table_[index];

    // 候補手の展開
    current_node.legal_moves = pos.generateAllMoves();
    current_node.child_num = (uint32_t)current_node.legal_moves.size();
    current_node.child_indices = std::vector<int32_t>(current_node.child_num, UctHashTable::NOT_EXPANDED);
    current_node.child_move_counts = std::vector<int32_t>(current_node.child_num, 0);

    // 現在のノードの初期化
    current_node.move_count = 0;
    current_node.evaled = false;
#ifdef USE_CATEGORICAL
    current_node.child_wins = std::vector<std::array<CalcType, BIN_SIZE>>(current_node.child_num);
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        current_node.value_dist[i] = 0.0;
        current_node.win_sum[i] = 0.0;
        for (int32_t j = 0; j < current_node.child_num; j++) {
            current_node.child_wins[j][i] = 0.0;
        }
    }
#else
    current_node.value_win = 0.0;
    current_node.win_sum = 0.0;
    current_node.child_wins = std::vector<float>(current_node.child_num, 0.0);
#endif

    //終局しているかどうかで条件分け
    //ノードを評価
    if (!pos.isFinish()) {
        evalNode(pos, index);
    } else {
        printf("ここには来ないはず\n");
        assert(false);
        //終了
        int32_t num = pos.score();
        double result;
        if (num == 0) {
            result = 0.5;
        } else if (num > 0) {
            result = 1.0;
        } else {
            result = 0.0;
        }
#ifdef USE_CATEGORICAL
        current_node.value_dist = onehotDist(result);
#else
        current_node.value_win = (CalcType)result;
#endif
        current_node.evaled = true;
    }

    return index;
}

void MCTSearcher::evalNode(Position& pos, Index index) {
    auto& current_node = hash_table_[index];
    std::vector<float> legal_move_policy(current_node.child_num);

    //Policyの計算
    if (current_node.child_num != 1) {
        auto policy_and_value = pos.policy();

        for (int32_t i = 0; i < current_node.child_num; i++) {
            legal_move_policy[i] = policy_and_value[current_node.legal_moves[i].toLabel()];
        }

        //softmax分布にする
        current_node.nn_rates = softmax(legal_move_policy);
    } else {
        //1手だけだから計算するまでもない
        current_node.nn_rates.assign(1, 1.0);
    }

    //ノードの値を計算
#ifdef USE_CATEGORICAL
    current_node.value_dist = pos.valueDist();
#else
    current_node.value_win = (CalcType)sigmoid(pos.valueScoreForTurn(), 1.0);
#endif
    current_node.evaled = true;
}

bool MCTSearcher::isTimeOver() {
    //時間のチェック
    auto now_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - start_);
    return (elapsed.count() >= shared_data.limit_msec - usi_option.byoyomi_margin);
}

bool MCTSearcher::shouldStop() {
    if (isTimeOver()) {
        return true;
    }

    // 探索回数が最も多い手と次に多い手を求める
    int32_t max1 = 0, max2 = 0;
    for (auto e : hash_table_[current_root_index_].child_move_counts) {
        if (e > max1) {
            max2 = max1;
            max1 = e;
        } else if (e > max2) {
            max2 = e;
        }
    }

    // 残りの探索を全て次善手に費やしても最善手を超えられない場合は探索を打ち切る
    return (max1 - max2) > (usi_option.playout_limit - playout_num);
}

std::vector<Move> MCTSearcher::getPV() const {
    std::vector<Move> pv;
    for (Index curr_node_index = current_root_index_; curr_node_index != UctHashTable::NOT_EXPANDED && hash_table_[curr_node_index].child_num != 0; ) {
        const auto& child_move_counts = hash_table_[curr_node_index].child_move_counts;
        Index next_index = (int32_t)(std::max_element(child_move_counts.begin(), child_move_counts.end()) - child_move_counts.begin());
        pv.push_back(hash_table_[curr_node_index].legal_moves[next_index]);
        curr_node_index = hash_table_[curr_node_index].child_indices[next_index];
    }

    return pv;
}

void MCTSearcher::printUSIInfo() const {
    const auto& current_node = hash_table_[current_root_index_];

    //探索にかかった時間を求める
    auto finish_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_);

    const auto& child_move_counts = current_node.child_move_counts;
    int32_t selected_index = (int32_t)(std::max_element(child_move_counts.begin(), child_move_counts.end()) - child_move_counts.begin());

    //選択した着手の勝率の算出
#ifdef USE_CATEGORICAL
    double best_wp = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        double v = current_node.child_wins[selected_index][i] / child_move_counts[selected_index];
        best_wp += VALUE_WIDTH * (0.5 + i) * v;
    }
#else
    auto best_wp = (child_move_counts[selected_index] == 0 ? 0.0
        : current_node.child_wins[selected_index] / child_move_counts[selected_index]);
#endif
    assert(0.0 <= best_wp && best_wp <= 1.0);

    //勝率を評価値に変換
    int32_t cp = inv_sigmoid(best_wp, CP_GAIN);

    printf("info nps %d time %d nodes %d hashfull %d score cp %d pv ",
        (int)(current_node.move_count * 1000 / std::max((long long)elapsed.count(), 1LL)),
        (int)(elapsed.count()),
        current_node.move_count,
        (int)(hash_table_.getUsageRate() * 1000),
        cp);

    auto pv = getPV();
    for (auto m : pv) {
        std::cout << m << " ";
    }
    std::cout << std::endl;
}

std::vector<double> MCTSearcher::dirichletDistribution(int32_t k, double alpha) {
    static std::random_device seed;
    static std::default_random_engine engine(seed());
    static constexpr double eps = 0.000000001;
    std::gamma_distribution<double> gamma(alpha, 1.0);
    std::vector<double> dirichlet(k);
    double sum = 0.0;
    for (int32_t i = 0; i < k; i++) {
        sum += (dirichlet[i] = std::max(gamma(engine), eps));
    }
    for (int32_t i = 0; i < k; i++) {
        dirichlet[i] /= sum;
    }
    return dirichlet;
}

#ifdef USE_CATEGORICAL
int32_t MCTSearcher::selectMaxUcbChild(const UctHashEntry & current_node, double curr_best_winrate) {
    const auto& child_move_counts = current_node.child_move_counts;

    // ucb = Q(s, a) + U(s, a)
    // Q(s, a) = W(s, a) / N(s, a)
    // U(s, a) = C_PUCT * P(s, a) * sqrt(sum_b(B(s, b)) / (1 + N(s, a))
    constexpr double C_PUCT = 1.0;

    int32_t max_index = -1;
    double max_value = INT_MIN;
    for (int32_t i = 0; i < current_node.child_num; i++) {
        double Q = 0.0;
        for (int32_t j = (int32_t)(curr_best_winrate * BIN_SIZE); j < BIN_SIZE; j++) {
            Q += (child_move_counts[i] == 0 ? 0.0 : current_node.child_wins[i][j] / child_move_counts[i]);
        }
        double U = std::sqrt(current_node.move_count + 1) / (child_move_counts[i] + 1);
        double ucb = Q + C_PUCT * current_node.nn_rates[i] * U;

        //詰みだったらそれを選べばいいだろう
        if (Q == 1.0) {
            return i;
        }
        if (ucb > max_value) {
            max_value = ucb;
            max_index = i;
        }
    }
    assert(0 <= max_index && max_index < (int32_t)current_node.child_num);
    return max_index;
}

#else
int32_t MCTSearcher::selectMaxUcbChild(const UctHashEntry & current_node) {
    const auto& child_move_counts = current_node.child_move_counts;

    // ucb = Q(s, a) + U(s, a)
    // Q(s, a) = W(s, a) / N(s, a)
    // U(s, a) = C_PUCT * P(s, a) * sqrt(sum_b(B(s, b)) / (1 + N(s, a))
    constexpr double C_PUCT = 1.0;

    int32_t max_index = -1;
    double max_value = INT_MIN;
    for (int32_t i = 0; i < current_node.child_num; i++) {
        double Q = (child_move_counts[i] == 0 ? 0.5 : current_node.child_wins[i] / child_move_counts[i]);
        double U = std::sqrt(current_node.move_count + 1) / (child_move_counts[i] + 1);
        double ucb = Q + C_PUCT * current_node.nn_rates[i] * U;

        //詰みだったらそれを選べばいいだろう
        if (Q == 1.0) {
            return i;
        }
        if (ucb > max_value) {
            max_value = ucb;
            max_index = i;
        }
    }
    assert(0 <= max_index && max_index < (int32_t)current_node.child_num);
    return max_index;
}
#endif

#endif