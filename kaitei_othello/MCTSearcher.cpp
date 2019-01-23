#include"MCTSearcher.hpp"
#include"shared_data.hpp"
#include"network.hpp"
#include"usi_options.hpp"
#include"operate_params.hpp"
#include<stack>

#ifdef USE_MCTS

std::pair<Move, TeacherType> MCTSearcher::thinkForGenerateLearnData(Position& root, bool add_noise) {
    //�v�l�J�n���Ԃ��Z�b�g
    start_ = std::chrono::steady_clock::now();

    //�Â��n�b�V�����폜
    hash_table_.deleteOldHash(root, add_noise);

    //���[�g�m�[�h�̓W�J
    current_root_index_ = expandNode(root);
    auto& current_node = hash_table_[current_root_index_];

    //NULL_MOVE�����Ȃ炷���Ԃ�
    if (current_node.moves_size == 1 && current_node.moves[0] == NULL_MOVE) {
        return { NULL_MOVE, TeacherType() };
    }

    if (add_noise) {
        //�m�C�Y��������
        //Alpha Zero�̘_���Ɠ����f�B���N���m�C�Y
        constexpr double epsilon = 0.25;
        auto dirichlet = dirichletDistribution(current_node.moves_size, 0.05);
        for (int32_t i = 0; i < current_node.moves_size; i++) {
            current_node.policy[i] = (CalcType)((1.0 - epsilon) * current_node.policy[i] + epsilon * dirichlet[i]);
        }
    }

    //������
    playout_num = 0;

    //�v���C�A�E�g���J��Ԃ�
    //�T���񐔂�臒l�𒴂���A�܂��͒T�����ł��؂�ꂽ�烋�[�v�𔲂���
    while (playout_num < usi_option.playout_limit) {
        //�T���񐔂�1�񑝂₷
        playout_num++;

        //1��v���C�A�E�g
        uctSearch(root, current_root_index_);
        //onePlay(root);

        //�T����ł��؂邩�m�F
        if (shouldStop() || !hash_table_.hasEnoughSize()) {
            break;
        }
    }

    //std::cout << "playout_num = " << playout_num << std::endl;
    //auto end = std::chrono::steady_clock::now();
    //auto elapsed = end - start_;
    //std::cout << "time = " << elapsed.count() << std::endl;

    const auto& N = current_node.N;

    //for debug
    //root.print();
    //auto moves_size = current_node.moves_size;
    //auto root_moves = current_node.moves;
    //for (int32_t i = 0; i < moves_size; i++) {
    //    printf("%3d: sum_N = %6d, nn_rate = %.5f, win_rate = %7.5f, ", i, N[i],
    //        current_node.policy[i], (N[i] > 0 ? expOfValueDist(current_node.W[i]) / N[i] : 0));
    //    root_moves[i].print();
    //}
    
    // �K��񐔍ő�̎��I������
    int32_t best_index = (int32_t)(std::max_element(N.begin(), N.end()) - N.begin());

    //�I����������̏����̎Z�o
#ifdef USE_CATEGORICAL
    double best_wp = expOfValueDist(current_node.W[best_index]) / current_node.N[best_index];
#else
    auto best_wp = (N[best_index] == 0 ? 0.0
        : current_node.W[best_index] / N[best_index]);
#endif
    assert(0.0 <= best_wp && best_wp <= 1.0);

    TeacherType teacher(OUTPUT_DIM, 0.0);

    //�K��񐔂Ɋ�Â������z�𓾂�
    std::vector<double> distribution(current_node.moves_size);
    for (int32_t i = 0; i < current_node.moves_size; i++) {
        distribution[i] = (double)N[i] / current_node.sum_N;
        assert(0.0 <= distribution[i] && distribution[i] <= 1.0);

        //���z�����t�f�[�^�ɃZ�b�g
        teacher[current_node.moves[i].toLabel()] = (CalcType)distribution[i];
    }

    //value�̃Z�b�g
#ifdef USE_CATEGORICAL
    for (int32_t i = 0; i < current_node.moves_size; i++) {
        if (current_node.N[i] == 0) {
            //distribution[i] == 0�̂͂��Ȃ̂Ōv�Z����Ӗ����Ȃ�
            assert(distribution[i] == 0.0);
            continue;
        } 

        double exp = expOfValueDist(current_node.W[i]) / current_node.N[i];
        assert(0.0 <= exp && exp <= 1.0);

        //���Ғl�̂Ƃ���ɓ�������
        teacher[POLICY_DIM + valueToIndex(exp)] += (CalcType)distribution[i];
    }
#else
    teacher[POLICY_DIM] = (CalcType)best_wp;
#endif

    //�őP��
    //�萔���w��ȉ��������ꍇ�͖K��񐔂̕��z���烉���_���ɑI��
    Move best_move = (root.turn_number() < usi_option.random_turn
        ? current_node.moves[randomChoise(distribution)]
        : current_node.moves[best_index]);
    best_move.score = (Score)best_wp;
    
    //best_index�̕��z��\��
    //CalcType value = 0.0;
    //double sum_p = 0.0;
    //for (int32_t i = 0; i < BIN_SIZE; i++) {
    //    double p = current_node.child_wins[best_index][i] / current_node.N[best_index];
    //    printf("p[%f] = %f ", VALUE_WIDTH * (0.5 + i), p);
    //    for (int32_t j = 0; j < (int32_t)(p / 0.02); j++) {
    //        std::cout << "*";
    //    }
    //    std::cout << std::endl;
    //    value += (CalcType)(VALUE_WIDTH * (0.5 + i) * p);
    //    sum_p += p;
    //}
    //printf("value = %f, sum_p = %f\n", value, sum_p);

    return { best_move, teacher };
}

#ifdef USE_CATEGORICAL
std::array<CalcType, BIN_SIZE> MCTSearcher::uctSearch(Position & pos, Index current_index) {
#else
CalcType MCTSearcher::uctSearch(Position & pos, Index current_index) {
#endif
    auto& current_node = hash_table_[current_index];

    auto& child_indices = current_node.child_indices;

    // UCB�l���ő�̎�����߂�
    auto next_index = selectMaxUcbChild(current_node);

    // �I�񂾎�𒅎�
    pos.doMove(current_node.moves[next_index]);

#ifdef USE_CATEGORICAL
    std::array<CalcType, BIN_SIZE> result;
#else
    CalcType result;
#endif
    // �m�[�h�̓W�J�̊m�F
    if (pos.isFinish()) {
        //�I��
#ifdef USE_CATEGORICAL
        result = onehotDist(pos.resultForTurn());
        std::reverse(result.begin(), result.end());
#else
        result = (CalcType)(1.0 - pos.resultForTurn());
#endif
    } else if (child_indices[next_index] == UctHashTable::NOT_EXPANDED) {
        // �m�[�h�̓W�J
        auto index = expandNode(pos);
        child_indices[next_index] = index;
#ifdef USE_CATEGORICAL
        result = hash_table_[index].value;
        std::reverse(result.begin(), result.end());
#else
        result = 1.0f - hash_table_[index].value;
#endif
    } else {
        // ��Ԃ����ւ���1��[���ǂ�
#ifdef USE_CATEGORICAL
        result = uctSearch(pos, child_indices[next_index]);
        std::reverse(result.begin(), result.end());
#else
        result = 1.0f - uctSearch(pos, child_indices[next_index]);
#endif
    }

    // �T�����ʂ̔��f
    current_node.sum_N++;
    current_node.W[next_index] += result;
    current_node.N[next_index]++;

    // ���߂�
    pos.undo();

    return result;
}

Index MCTSearcher::expandNode(Position& pos) {
    auto index = hash_table_.findSameHashIndex(pos.hash_value(), pos.turn_number());

    // �����悪���m�ł���΂����Ԃ�
    if (index != hash_table_.size()) {
        return index;
    }

    // ��̃C���f�b�N�X��T��
    index = hash_table_.searchEmptyIndex(pos.hash_value(), pos.turn_number());

    auto& current_node = hash_table_[index];

    // ����̓W�J
    current_node.moves = pos.generateAllMoves();
    current_node.moves_size = (uint32_t)current_node.moves.size();
    current_node.child_indices = std::vector<int32_t>(current_node.moves_size, UctHashTable::NOT_EXPANDED);
    current_node.N = std::vector<int32_t>(current_node.moves_size, 0);

    // ���݂̃m�[�h�̏�����
    current_node.sum_N = 0;
    current_node.evaled = false;
#ifdef USE_CATEGORICAL
    current_node.W = std::vector<std::array<CalcType, BIN_SIZE>>(current_node.moves_size);
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        current_node.value[i] = 0.0;
        for (int32_t j = 0; j < current_node.moves_size; j++) {
            current_node.W[j][i] = 0.0;
        }
    }
#else
    current_node.value = 0.0;
    current_node.W = std::vector<float>(current_node.moves_size, 0.0);
#endif

    //�m�[�h��]��
    evalNode(pos, index);

    return index;
}

void MCTSearcher::evalNode(Position& pos, Index index) {
    auto& current_node = hash_table_[index];
    std::vector<float> legal_move_policy(current_node.moves_size);

    //Policy�̌v�Z
    if (current_node.moves_size != 1) {
        auto policy_score = pos.policyScore();

        for (int32_t i = 0; i < current_node.moves_size; i++) {
            legal_move_policy[i] = policy_score[current_node.moves[i].toLabel()];
        }

        //softmax���z�ɂ���
        current_node.policy = softmax(legal_move_policy);
    } else {
        //1�肾��������v�Z����܂ł��Ȃ�
        current_node.policy.assign(1, 1.0);
    }

    //�m�[�h�̒l���v�Z
#ifdef USE_CATEGORICAL
    current_node.value = pos.valueDist();
#else
    current_node.value = (CalcType)pos.valueForTurn();
#endif
    current_node.evaled = true;
}

bool MCTSearcher::isTimeOver() {
    //���Ԃ̃`�F�b�N
    auto now_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - start_);
    return (elapsed.count() >= shared_data.limit_msec);
}

bool MCTSearcher::shouldStop() {
    if (isTimeOver()) {
        return true;
    }

    return false;

    // �T���񐔂��ł�������Ǝ��ɑ���������߂�
    int32_t max1 = 0, max2 = 0;
    for (auto e : hash_table_[current_root_index_].N) {
        if (e > max1) {
            max2 = max1;
            max1 = e;
        } else if (e > max2) {
            max2 = e;
        }
    }

    // �c��̒T����S�Ď��P��ɔ�₵�Ă��őP��𒴂����Ȃ��ꍇ�͒T����ł��؂�
    return (max1 - max2) > (usi_option.playout_limit - playout_num);
}

std::vector<Move> MCTSearcher::getPV() const {
    std::vector<Move> pv;
    for (Index curr_node_index = current_root_index_; curr_node_index != UctHashTable::NOT_EXPANDED && hash_table_[curr_node_index].moves_size != 0; ) {
        const auto& N = hash_table_[curr_node_index].N;
        Index next_index = (int32_t)(std::max_element(N.begin(), N.end()) - N.begin());
        pv.push_back(hash_table_[curr_node_index].moves[next_index]);
        curr_node_index = hash_table_[curr_node_index].child_indices[next_index];
    }

    return pv;
}

void MCTSearcher::printUSIInfo() const {
    const auto& current_node = hash_table_[current_root_index_];

    //�T���ɂ����������Ԃ����߂�
    auto finish_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_);

    const auto& N = current_node.N;
    int32_t selected_index = (int32_t)(std::max_element(N.begin(), N.end()) - N.begin());

    //�I����������̏����̎Z�o
#ifdef USE_CATEGORICAL
    double best_wp = expOfValueDist(current_node.W[selected_index]) / N[selected_index];
#else
    auto best_wp = (N[selected_index] == 0 ? 0.0
        : current_node.W[selected_index] / N[selected_index]);
#endif
    assert(0.0 <= best_wp && best_wp <= 1.0);

    //������]���l�ɕϊ�
    int32_t cp = inv_sigmoid(best_wp, CP_GAIN);

    printf("info nps %d time %d nodes %d hashfull %d score cp %d pv ",
        (int)(current_node.sum_N * 1000 / std::max((long long)elapsed.count(), 1LL)),
        (int)(elapsed.count()),
        current_node.sum_N,
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

int32_t MCTSearcher::selectMaxUcbChild(const UctHashEntry & current_node) {
    const auto& N = current_node.N;

    // ucb = Q(s, a) + U(s, a)
    // Q(s, a) = W(s, a) / N(s, a)
    // U(s, a) = C_PUCT * P(s, a) * sqrt(sum_b(B(s, b)) / (1 + N(s, a))
    constexpr double C_PUCT = 1.0;

    int32_t max_index = -1;
    double max_value = INT_MIN;

#ifdef USE_CATEGORICAL
    int32_t best_index = (int32_t)(std::max_element(N.begin(), N.end()) - N.begin());
    double best_wp = (current_node.N[best_index] == 0 ? 0.0:
        expOfValueDist(current_node.W[best_index]) / current_node.N[best_index]);
#endif

    for (int32_t i = 0; i < current_node.moves_size; i++) {
#ifdef USE_CATEGORICAL
        double Q = 0.0;
        if (N[i] == 0) {
            Q = 0.5;
        } else {
            ////(1)���ʂɊ��Ғl���v�Z����
            //Q = expOfValueDist(current_node.W[i]) / N[i];

            ////(2)���U��(1)�ɉ�����
            //auto e = Q;
            //for (int32_t j = 0; j < BIN_SIZE; j++) {
            //    Q += pow(VALUE_WIDTH * (0.5 + j) - e, 2) *
            //        (N[i] == 0 ? 0.0 : current_node.child_wins[i][j] / N[i]);
            //}

            //(3)��l�𒴂���m��(��Ď�@)
            for (int32_t j = std::min(valueToIndex(best_wp) + 1, BIN_SIZE - 1); j < BIN_SIZE; j++) {
                Q += current_node.W[i][j] / N[i];
            }
        }
        assert(-0.01 <= Q && Q <= 1.01);
#else
        double Q = (N[i] == 0 ? 0.5 : current_node.W[i] / N[i]);
#endif
        constexpr double C_base = 19652.0;
        constexpr double C_init = 1.25;
        double C = (std::log((current_node.sum_N + C_base + 1) / C_base) + C_init) / 2;
        
        double U = std::sqrt(current_node.sum_N + 1) / (N[i] + 1);
        double ucb = Q + C * current_node.policy[i] * U;
        if (ucb > max_value) {
            max_value = ucb;
            max_index = i;
        }
    }
    assert(0 <= max_index && max_index < (int32_t)current_node.moves_size);
    return max_index;
}

void MCTSearcher::onePlay(Position& pos) {
    std::stack<Index> indices;
    std::stack<int32_t> actions;

    auto index = current_root_index_;

    //���W�J�̋ǖʂɎ���܂őJ�ڂ��J��Ԃ�
    while (index != UctHashTable::NOT_EXPANDED) {
        //��Ԃ��L�^
        indices.push(index);

        //�I��
        auto action = selectMaxUcbChild(hash_table_[index]);

        //������s�����L�^
        actions.push(action);

        //�J��
        pos.doMove(hash_table_[index].moves[action]);

        //index�X�V
        index = hash_table_[index].child_indices[action];
    }

    //���̋ǖʂ�W�J�E�]��
    index = expandNode(pos);
    auto result = hash_table_[index].value;
    hash_table_[indices.top()].child_indices[actions.top()] = index;

    //�o�b�N�A�b�v
    while (!actions.empty()) {
        pos.undo();
        index = indices.top();
        indices.pop();

        auto action = actions.top();
        actions.pop();

        //��Ԃ��ς���Ă���̂Ŕ��]
#ifdef USE_CATEGORICAL
        std::reverse(result.begin(), result.end());
#else
        result = MAX_SCORE + MIN_SCORE - result;
#endif

        // �T�����ʂ̔��f
        hash_table_[index].W[action] += result;
        hash_table_[index].sum_N++;
        hash_table_[index].N[action]++;
    }
}


#endif