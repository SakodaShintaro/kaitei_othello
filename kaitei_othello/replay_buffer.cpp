#include"position.hpp"
#include"operate_params.hpp"
#include"alphazero_trainer.hpp"

std::vector<std::pair<std::array<int64_t, 3>, TeacherType>> ReplayBuffer::makeBatch(int64_t batch_size) {
    std::vector<std::pair<std::array<int64_t, 3>, TeacherType>> batch;
    batch.reserve(batch_size);
    static std::mt19937 engine(0);
    mutex.lock();

    while ((int64_t)buffer_.size() < WAIT_LIMIT_SIZE * batch_size) {
        mutex.unlock();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        mutex.lock();
    }

    //�f�[�^��max_size_�𒴉߂��Ă����猸�炷
    if ((int64_t)buffer_.size() > MAX_STACK_SIZE) {
        buffer_.erase(buffer_.begin(), buffer_.begin() + MAX_STACK_SIZE - buffer_.size());
    }

    //�����_���Ƀf�[�^�����
    std::uniform_int_distribution<int64_t> dist(0, buffer_.size() - 1);
    for (int64_t i = 0; i < batch_size; i++) {
        batch.push_back(buffer_[dist(engine)]);
    }
    mutex.unlock();
    return batch;
}

void ReplayBuffer::push(Game& game) {
    Position pos(*eval_params);

    //�܂��͍ŏI�ǖʂ܂œ�����
    for (auto move : game.moves) {
        pos.doMove(move);
    }

    assert(Game::RESULT_WHITE_WIN <= game.result && game.result <= Game::RESULT_BLACK_WIN);

    //��肩�猩������,���z.�w���ړ����ςœ������Ă���.�ŏ��͌��ʂɂ���ď�����(0 or 0.5 or 1)
    double win_rate_for_black = game.result;

    for (int32_t i = (int32_t)game.moves.size() - 1; i >= 0; i--) {
        //i�Ԗڂ̎w���肪�Ή�����̂�1��߂����ǖ�
        pos.undo();

        if (game.moves[i] == NULL_MOVE) {
            //�p�X��������w�K���΂�
            continue;
        }

#ifdef USE_CATEGORICAL
        //��Ԃ��猩�����z�𓾂�
        auto teacher_dist = onehotDist(pos.color() == BLACK ? win_rate_for_black : 1.0 - win_rate_for_black);

        //teacher�ɃR�s�[����
        std::copy(teacher_dist.begin(), teacher_dist.end(), game.teachers[i].begin() + POLICY_DIM);
#else
        //teacher�ɃR�s�[����
        game.teachers[i][POLICY_DIM] = (CalcType)(pos.color() == BLACK ? win_rate_for_black : 1.0 - win_rate_for_black);
#endif
        //�T�����ʂ��肩�猩���l�ɕϊ�
        double curr_win_rate = (pos.color() == BLACK ? game.moves[i].score : 1.0 - game.moves[i].score);

        //����
        win_rate_for_black = LAMBDA * win_rate_for_black + (1.0 - LAMBDA) * curr_win_rate;

        //�X�^�b�N�ɋl�߂�
        mutex.lock();
        buffer_.push_back({ pos.data(), game.teachers[i] });
        mutex.unlock();
    }
}

void ReplayBuffer::clear() {
    mutex.lock();
    buffer_.clear();
    buffer_.reserve(MAX_STACK_SIZE);
    mutex.unlock();
}