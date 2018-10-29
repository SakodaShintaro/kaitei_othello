#include"base_trainer.hpp"
#include"network.hpp"

//#define PRINT_DEBUG

#ifdef USE_NN
std::array<double, 2> BaseTrainer::addGrad(EvalParams<LearnEvalType>& grad, Position& pos, TeacherType teacher) {
    assert(teacher.size() == OUTPUT_DIM);

    const auto input = pos.makeFeatures();
    const auto& params = pos.evalParams();
    const Vec input_vec = Eigen::Map<const Vec>(input.data(), input.size());
    const Vec u0 = params.w[0] * input_vec + params.b[0];
    const Vec z0 = Network::activationFunction(u0);

    //Policy
    auto y = softmax(pos.policy());

    //Policy‚Ì‘¹¸
    double policy_loss = 0.0;
    for (int32_t i = 0; i < POLICY_DIM; i++) {
        policy_loss += crossEntropy(y[i], teacher[i]);
    }

    //Value‚Ì‘¹¸
#ifdef USE_CATEGORICAL
    auto v = pos.valueDist();

    double value_loss = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        value_loss += crossEntropy(v[i], teacher[POLICY_DIM + i]);
#ifdef PRINT_DEBUG
        std::cout << v[i] << " " << teacher[POLICY_DIM + i] << std::endl;
#endif
    }
#else
    const auto win_rate = sigmoid(pos.valueScoreForTurn(), 1.0);
    double value_loss = binaryCrossEntropy(win_rate, teacher[POLICY_DIM]);
#endif

    //Policy‚ÌŒù”z
    Vec delta_o(OUTPUT_DIM);
#ifdef PRINT_DEBUG
    double abs_sum = 0.0;
    double abs_max = 0.0;
#endif
    for (int32_t i = 0; i < POLICY_DIM; i++) {
        delta_o(i) = y[i] - teacher[i];
#ifdef PRINT_DEBUG
        abs_sum += std::abs(delta_o(i));
        abs_max = std::max(abs_max, (double)(std::abs(delta_o(i))));
#endif
    }
    //Value‚ÌŒù”z
#ifdef USE_CATEGORICAL
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        delta_o(POLICY_DIM + i) = (CalcType)(VALUE_COEFF * (v[i] - teacher[POLICY_DIM + i]));
    }
#else
    delta_o(POLICY_DIM) = (CalcType)(VALUE_COEFF * (win_rate - teacher[POLICY_DIM]));
#endif

#ifdef PRINT_DEBUG
    std::cout << "abs_sum = " << abs_sum << std::endl;
    std::cout << "abs_max = " << abs_max << std::endl;
    std::cout << "value   = " << delta_o(POLICY_DIM) << "\n" << std::endl;

    pos.print();
    for (auto move : pos.generateAllMoves()) {
        auto index = move.toLabel();
        std::cout << y[index] << " " << teacher[index] << " ";
        move.print();
    }
#endif

    //‹t“`”d
    Vec delta_h = Network::d_activationFunction(u0).array() * (params.w[1].transpose() * delta_o).array();

    grad.w[1] += delta_o * z0.transpose();
    grad.b[1] += delta_o;

    grad.w[0] += delta_h * input_vec.transpose();
    grad.b[0] += delta_h;

    return { policy_loss, value_loss };
}
#else
double BaseTrainer::addGrad(EvalParams<LearnEvalType>& grad, Position& pos, TeacherType teacher) {
    //•]‰¿’l‚ğæ“¾
    int32_t score = (int32_t)pos.scoreForTurn();

    //‘¹¸‚ğŒvZ
    double win_y = sigmoid(score, CP_GAIN);
    double loss = binaryCrossEntropy(win_y, teacher);

    //Œù”z‚Ì•Ï‰»—Ê‚ğŒvZ
    double grad_delta = win_y - teacher;
    double grad_delta2 = grad_delta * d_sigmoid(score, CP_GAIN) / CP_GAIN * 4;

#ifdef PRINT_DEBUG
    std::cout << "y = " << win_y << ", t = " << teacher << std::endl;
    std::cout << grad_delta << std::endl;
    std::cout << grad_delta2 << std::endl;
#endif

    //Œãè‚Ì‚ÍŒù”z‚Ì•„†‚ğ”½“]‚³‚¹‚é
    updateGradient(grad, pos.features(), (pos.color() == BLACK ? grad_delta : -grad_delta));

    return loss;
}
#endif

#ifdef USE_NN
void BaseTrainer::verifyAddGrad(Position & pos, TeacherType teacher) {
    auto grad_bp = std::make_unique<EvalParams<LearnEvalType>>();
    auto loss = addGrad(*grad_bp, pos, teacher);

    constexpr CalcType eps = 0.001f;
    std::cout << std::fixed << std::setprecision(15);

    //’l‚ğ•Ï‚¦‚¸‚É‡“`”d‚µ‚½‚Æ‚«‚Ì‘¹¸
    double loss1 = 0.0;
    auto y1 = softmax(pos.policy());
    auto value1 = sigmoid(pos.valueScoreForTurn(), 1.0);
    std::cout << "value1 = " << value1 << std::endl;
    for (int32_t l = 0; l < POLICY_DIM; l++) {
        loss1 += crossEntropy(y1[l], teacher[l]);
    }
    loss1 += binaryCrossEntropy(value1, teacher[POLICY_DIM]);

    for (int32_t i = 0; i < LAYER_NUM; i++) {
        for (int32_t j = 0; j < MATRIX_SIZE[i][0]; j++) {
            for (int32_t k = 0; k < MATRIX_SIZE[i][1]; k++) {

                eval_params->w[i](j, k) += eps;
                double loss2 = 0.0;
                auto y2 = softmax(pos.policy());
                auto value2 = sigmoid(pos.valueScoreForTurn(), 1.0);
                for (int32_t l = 0; l < POLICY_DIM; l++) {
                    loss2 += crossEntropy(y2[l], teacher[l]);
                }
                loss2 += binaryCrossEntropy(value2, teacher[POLICY_DIM]);
                eval_params->w[i](j, k) -= eps;

                double grad = (loss2 - loss1) / eps;

                if (abs(grad - grad_bp->w[i](j, k)) >= 0.005) {
                    printf("Œù”z‚ª‚¨‚©‚µ‚¢\n");
                    std::cout << "(i, j, k) = (" << i << ", " << j << ", " << k << ")" << std::endl;
                    std::cout << "loss    = " << loss[0] + loss[1]  << std::endl;
                    std::cout << "loss1   = " << loss1 << std::endl;
                    std::cout << "loss2   = " << loss2 << std::endl;
                    std::cout << "grad    = " << grad << std::endl;
                    std::cout << "grad_bp = " << grad_bp->w[i](j, k) << std::endl;
                }
            }

            eval_params->b[i](j) += eps;
            double loss2 = 0.0;
            auto y2 = softmax(pos.policy());
            auto value2 = sigmoid(pos.valueScoreForTurn(), 1.0);
            for (int32_t l = 0; l < POLICY_DIM; l++) {
                loss2 += crossEntropy(y2[l], teacher[l]);
            }
            loss2 += binaryCrossEntropy(value2, teacher[POLICY_DIM]);
            eval_params->b[i](j) -= eps;

            double grad = (loss2 - loss1) / eps;

            if (std::abs(grad - grad_bp->b[i](j)) >= 0.005) {
                printf("Œù”z‚ª‚¨‚©‚µ‚¢\n");
                std::cout << "(i, j) = (" << i << ", " << j << ")" << std::endl;
                std::cout << "loss    = " << loss[0] + loss[1] << std::endl;
                std::cout << "loss1   = " << loss1 << std::endl;
                std::cout << "loss2   = " << loss2 << std::endl;
                std::cout << "grad    = " << grad << std::endl;
                std::cout << "grad_bp = " << grad_bp->b[i](j) << std::endl;
                if (j < POLICY_DIM) {
                    std::cout << "y = " << y1[j] << std::endl;
                    std::cout << "t = " << teacher[j] << std::endl;
                } else {
                    std::cout << "v = " << value1 << std::endl;
                    std::cout << "t = " << teacher[j] << std::endl;
                }
            }
        }
    }
}
#endif

#ifndef USE_NN
void BaseTrainer::updateGradient(EvalParams<LearnEvalType>& grad, const Features& features, const LearnEvalType delta) {
    //features‚Éo‚Ä‚­‚é“Á’¥—Ê‚ÉŠÖ‚í‚éŒù”z‚·‚×‚Ä‚ğdelta‚¾‚¯•Ï‚¦‚é
    int32_t c = (features.color == BLACK ? 1 : -1);

    const int32_t bk_sq  = SquareToNum[features.king_sq[BLACK]];
    const int32_t bk_sqr = SquareToNum[InvSquare[features.king_sq[BLACK]]];
    const int32_t wk_sq  = SquareToNum[features.king_sq[WHITE]];
    const int32_t wk_sqr = SquareToNum[InvSquare[features.king_sq[WHITE]]];

    std::array<LearnEvalType, 2> d1 = {  delta, c * delta };
    std::array<LearnEvalType, 2> d2 = { -delta, c * delta };

    for (uint32_t i = 0; i < PIECE_STATE_LIST_SIZE; i++) {
        const PieceState nor_i = features.piece_state_list[0][i];
        const PieceState inv_i = features.piece_state_list[1][i];

        //•’Ê
        grad.kkp[bk_sq][wk_sq][nor_i] += d1;

        //180“x”½“]
        grad.kkp[wk_sqr][bk_sqr][inv_i] += d2;

        //¶‰E”½“]
        grad.kkp[mirrorSqNum(bk_sq)][mirrorSqNum(wk_sq)][mirrorPieceState(nor_i)] += d1;
        
        //180“x‚©‚Â¶‰E”½“]
        grad.kkp[mirrorSqNum(wk_sqr)][mirrorSqNum(bk_sqr)][mirrorPieceState(inv_i)] += d2;
        
        for (uint32_t j = i + 1; j < PIECE_STATE_LIST_SIZE; j++) {
            const PieceState nor_j = features.piece_state_list[0][j];
            const PieceState inv_j = features.piece_state_list[1][j];

            //•’Ê
            grad.kpp[bk_sq ][nor_i][nor_j] += d1;
            grad.kpp[wk_sqr][inv_i][inv_j] += d2;

            //‡”Ô‹t
            grad.kpp[bk_sq ][nor_j][nor_i] += d1;
            grad.kpp[wk_sqr][inv_j][inv_i] += d2;

            //¶‰E”½“]
            grad.kpp[mirrorSqNum(bk_sq) ][mirrorPieceState(nor_i)][mirrorPieceState(nor_j)] += d1;
            grad.kpp[mirrorSqNum(wk_sqr)][mirrorPieceState(inv_i)][mirrorPieceState(inv_j)] += d2;
            
            //¶‰E”½“]‚Ì‡”Ô‹t
            grad.kpp[mirrorSqNum(bk_sq) ][mirrorPieceState(nor_j)][mirrorPieceState(nor_i)] += d1;
            grad.kpp[mirrorSqNum(wk_sqr)][mirrorPieceState(inv_j)][mirrorPieceState(inv_i)] += d2;
        }
    }
}
#endif

void BaseTrainer::updateParams(EvalParams<LearnEvalType>& params, const EvalParams<LearnEvalType>& grad) {
    if (OPTIMIZER_NAME == "SGD") {
        updateParamsSGD(params, grad);
    } else if (OPTIMIZER_NAME == "MOMENTUM") {
        updateParamsMomentum(params, grad, *pre_update_);
    } else {
        std::cerr << "Illigal Optimizer Name : " << OPTIMIZER_NAME << std::endl;
        assert(false);
    }
}

void BaseTrainer::updateParamsSGD(EvalParams<LearnEvalType>& params, const EvalParams<LearnEvalType>& grad) {
#ifdef USE_NN
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        params.w[i].array() -= LEARN_RATE * grad.w[i].array();
        params.b[i].array() -= LEARN_RATE * grad.b[i].array();
    }
#else
    for (int k1 = 0; k1 < SqNum; k1++) {
        for (int p1 = 0; p1 < PieceStateNum; p1++) {
            for (int p2 = 0; p2 < PieceStateNum; p2++) {
                for (int t = 0; t < ColorNum; t++) {
                    params.kpp[k1][p1][p2][t] -= (LearnEvalType)(LEARN_RATE * grad.kpp[k1][p1][p2][t]);
                }
            }
            for (int k2 = 0; k2 < SqNum; k2++) {
                for (int t = 0; t < ColorNum; t++) {
                    params.kkp[k1][k2][p1][t] -= (LearnEvalType)(LEARN_RATE * grad.kkp[k1][k2][p1][t]);
                }
            }
        }
    }
#endif
}

void BaseTrainer::updateParamsMomentum(EvalParams<LearnEvalType>& params, const EvalParams<LearnEvalType>& grad, EvalParams<LearnEvalType>& pre_update) {
#ifdef USE_NN
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        auto curr_update_w = LEARN_RATE * grad.w[i] + MOMENTUM_DECAY * pre_update.w[i];
        auto curr_update_b = LEARN_RATE * grad.b[i] + MOMENTUM_DECAY * pre_update.b[i];
        params.w[i].array() -= curr_update_w.array();
        params.b[i].array() -= curr_update_b.array();
        pre_update.w[i] = curr_update_w;
        pre_update.b[i] = curr_update_b;
    }
#else
    for (int k1 = 0; k1 < SqNum; k1++) {
        for (int p1 = 0; p1 < PieceStateNum; p1++) {
            for (int p2 = 0; p2 < PieceStateNum; p2++) {
                for (int t = 0; t < ColorNum; t++) {
                    LearnEvalType curr_update = LEARN_RATE * grad.kpp[k1][p1][p2][t] + MOMENTUM_DECAY * pre_update.kpp[k1][p1][p2][t];
                    params.kpp[k1][p1][p2][t] -= curr_update;
                    pre_update.kpp[k1][p1][p2][t] = curr_update;
                }
            }
            for (int k2 = 0; k2 < SqNum; k2++) {
                for (int t = 0; t < ColorNum; t++) {
                    LearnEvalType curr_update = LEARN_RATE * grad.kkp[k1][k2][p1][t] + MOMENTUM_DECAY * pre_update.kkp[k1][k2][p1][t];
                    params.kkp[k1][k2][p1][t] -= curr_update;
                    pre_update.kkp[k1][k2][p1][t] = curr_update;
                }
            }
        }
    }
#endif
}

void BaseTrainer::timestamp() {
    auto elapsed = std::chrono::steady_clock::now() - start_time_;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    auto minutes = seconds / 60;
    seconds %= 60;
    auto hours = minutes / 60;
    minutes %= 60;
    std::cout << std::setfill('0') << std::setw(3) << hours << ":"
        << std::setfill('0') << std::setw(2) << minutes << ":"
        << std::setfill('0') << std::setw(2) << seconds << "\t";
    log_file_ << std::setfill('0') << std::setw(3) << hours << ":"
        << std::setfill('0') << std::setw(2) << minutes << ":"
        << std::setfill('0') << std::setw(2) << seconds << "\t";
}

bool BaseTrainer::isLegalOptimizer() {
    return (OPTIMIZER_NAME == "SGD"
        || OPTIMIZER_NAME == "MOMENTUM");
}