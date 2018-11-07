#include"base_trainer.hpp"
#include"network.hpp"

//#define PRINT_DEBUG

std::array<double, 2> BaseTrainer::addGrad(EvalParams<LearnEvalType>& grad, Position& pos, TeacherType teacher) {
    assert(teacher.size() == OUTPUT_DIM);

    const auto input = pos.makeFeature();
    const auto& params = pos.evalParams();
    const Vec input_vec = Eigen::Map<const Vec>(input.data(), input.size());

    Vec u[LAYER_NUM];
    Vec x[LAYER_NUM];
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        x[i] = (i == 0 ? input_vec : Network::activationFunction(u[i - 1]));
        u[i] = params.w[i] * x[i] + params.b[i];
    }

    //Policy
    auto y = softmax(pos.policy());

    //Policy‚Ì‘¹Ž¸
    double policy_loss = 0.0;
    for (int32_t i = 0; i < POLICY_DIM; i++) {
        policy_loss += crossEntropy(y[i], teacher[i]);
    }

    //Value‚Ì‘¹Ž¸
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
    const auto v = pos.valueForTurn();
    double value_loss = binaryCrossEntropy(v, teacher[POLICY_DIM]);
#endif

    //Policy‚ÌŒù”z
    Vec delta(OUTPUT_DIM);
#ifdef PRINT_DEBUG
    double abs_sum = 0.0;
    double abs_max = 0.0;
#endif
    for (int32_t i = 0; i < POLICY_DIM; i++) {
        delta_o(i) = (CalcType)(POLICY_LOSS_COEFF * (y[i] - teacher[i]));
#ifdef PRINT_DEBUG
        abs_sum += std::abs(delta(i));
        abs_max = std::max(abs_max, (double)(std::abs(delta(i))));
#endif
    }
    //Value‚ÌŒù”z
#ifdef USE_CATEGORICAL
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        delta(POLICY_DIM + i) = (CalcType)(VALUE_LOSS_COEFF * (v[i] - teacher[POLICY_DIM + i]));
    }
#else
    delta_o(POLICY_DIM) = (CalcType)(VALUE_LOSS_COEFF * (v - teacher[POLICY_DIM]));
#endif

#ifdef PRINT_DEBUG
    std::cout << "abs_sum = " << abs_sum << std::endl;
    std::cout << "abs_max = " << abs_max << std::endl;

    pos.print();
    for (auto move : pos.generateAllMoves()) {
        auto index = move.toLabel();
        std::cout << "index = " << index << " y = " <<  y[index] << "  t = " << teacher[index] << " ";
        move.print();
    }
    std::cout << "value = " << pos.valueForTurn() << " t = " << teacher[POLICY_DIM] << std::endl;
#endif

    //‹t“`”d
    for (int32_t i = LAYER_NUM - 1; i >= 0; i--) {
        grad.w[i] += delta * x[i].transpose();
        grad.b[i] += delta;
        if (i == 0) {
            break;
        }
        delta = Network::d_activationFunction(u[i - 1]).array() * (params.w[i].transpose() * delta).array();
    }

    return { policy_loss, value_loss };
}

void BaseTrainer::verifyAddGrad(Position & pos, TeacherType teacher) {
    auto grad_bp = std::make_unique<EvalParams<LearnEvalType>>();
    auto loss = addGrad(*grad_bp, pos, teacher);

    constexpr CalcType eps = 0.001f;
    std::cout << std::fixed << std::setprecision(15);

    //’l‚ð•Ï‚¦‚¸‚É‡“`”d‚µ‚½‚Æ‚«‚Ì‘¹Ž¸
    double loss1 = 0.0;
    auto y1 = softmax(pos.policy());
    auto value1 = pos.valueForTurn();
    std::cout << "value1 = " << value1 << std::endl;
    for (int32_t l = 0; l < POLICY_DIM; l++) {
        loss1 += crossEntropy(y1[l], teacher[l]);
    }
    loss1 += binaryCrossEntropy(value1, teacher[POLICY_DIM]);

    for (int32_t i = 0; i < LAYER_NUM; i++) {
        for (int32_t j = 0; j < MATRIX_SIZE[i][0]; j++) {
            for (int32_t k = 0; k < MATRIX_SIZE[i][1]; k++) {

                eval_params->w[i](j, k) += eps;
                pos.resetCalc();
                double loss2 = 0.0;
                auto y2 = softmax(pos.policy());
                auto value2 = pos.valueForTurn();
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
            pos.resetCalc();
            double loss2 = 0.0;
            auto y2 = softmax(pos.policy());
            auto value2 = pos.valueForTurn();
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
            }
        }
    }
}

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
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        params.w[i].array() -= LEARN_RATE * grad.w[i].array();
        params.b[i].array() -= LEARN_RATE * grad.b[i].array();
    }
}

void BaseTrainer::updateParamsMomentum(EvalParams<LearnEvalType>& params, const EvalParams<LearnEvalType>& grad, EvalParams<LearnEvalType>& pre_update) {
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        auto curr_update_w = LEARN_RATE * grad.w[i] + MOMENTUM_DECAY * pre_update.w[i];
        auto curr_update_b = LEARN_RATE * grad.b[i] + MOMENTUM_DECAY * pre_update.b[i];
        params.w[i].array() -= curr_update_w.array();
        params.b[i].array() -= curr_update_b.array();
        pre_update.w[i] = curr_update_w;
        pre_update.b[i] = curr_update_b;
    }
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