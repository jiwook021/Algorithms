/**
 * @file SelfTuningPidRl.cpp
 * @brief Implementation of Self-Tuning PID via Reinforcement Learning.
 */
#include "SelfTuningPidRl.hpp"

namespace Rl {

// ======================== PidController ========================

PidController::PidController(double kp, double ki, double kd, double dt)
    : Kp_(kp), Ki_(ki), Kd_(kd), Dt_(dt), PreviousError_(0.0), Integral_(0.0) {}

void PidController::UpdateParameters(double kp, double ki, double kd) {
    std::lock_guard<std::mutex> lock(Mutex_);
    Kp_ = kp; Ki_ = ki; Kd_ = kd;
}

double PidController::Compute(double error) {
    std::lock_guard<std::mutex> lock(Mutex_);
    double pTerm = Kp_ * error;
    Integral_ += error * Dt_;
    const double MAX_INTEGRAL = 10.0 / (Ki_ > 0.01 ? Ki_ : 0.01);
    Integral_ = std::clamp(Integral_, -MAX_INTEGRAL, MAX_INTEGRAL);
    double iTerm = Ki_ * Integral_;
    double derivative = (error - PreviousError_) / Dt_;
    double dTerm = Kd_ * derivative;
    PreviousError_ = error;
    return std::clamp(pTerm + iTerm + dTerm, -10.0, 10.0);
}

void PidController::Reset() {
    std::lock_guard<std::mutex> lock(Mutex_);
    PreviousError_ = 0.0; Integral_ = 0.0;
}

std::vector<double> PidController::GetParameters() const {
    std::lock_guard<std::mutex> lock(Mutex_);
    return {Kp_, Ki_, Kd_};
}

// ======================== NeuralNetwork ========================

NeuralNetwork::NeuralNetwork(size_t inputSize, size_t hiddenSize, size_t outputSize, double lr)
    : InputSize_(inputSize), HiddenSize_(hiddenSize), OutputSize_(outputSize), LearningRate_(lr) {
    std::random_device rd;
    std::mt19937 gen(rd());
    double f1 = std::sqrt(2.0 / (inputSize + hiddenSize));
    double f2 = std::sqrt(2.0 / (hiddenSize + outputSize));
    std::normal_distribution<double> dist(0.0, 0.1);
    W1_.resize(inputSize, std::vector<double>(hiddenSize));
    W2_.resize(hiddenSize, std::vector<double>(outputSize));
    B1_.resize(hiddenSize, 0.0);
    B2_.resize(outputSize, 0.0);
    Hidden_.resize(hiddenSize);
    for (auto& row : W1_) for (auto& w : row) w = dist(gen) * f1;
    for (auto& row : W2_) for (auto& w : row) w = dist(gen) * f2;
}

std::vector<double> NeuralNetwork::Forward(const std::vector<double>& input) {
    std::lock_guard<std::mutex> lock(Mutex_);
    std::vector<double> scaledInput = input;
    for (auto& v : scaledInput) v = std::clamp(v, -10.0, 10.0);
    Hidden_.resize(HiddenSize_);
    for (size_t j = 0; j < HiddenSize_; ++j) {
        Hidden_[j] = B1_[j];
        for (size_t i = 0; i < InputSize_; ++i) Hidden_[j] += scaledInput[i] * W1_[i][j];
        Hidden_[j] = std::max(0.0, Hidden_[j]);
    }
    std::vector<double> output(OutputSize_);
    for (size_t k = 0; k < OutputSize_; ++k) {
        output[k] = B2_[k];
        for (size_t j = 0; j < HiddenSize_; ++j) output[k] += Hidden_[j] * W2_[j][k];
        output[k] = std::clamp(output[k], -1.0, 1.0);
    }
    return output;
}

double NeuralNetwork::Train(const std::vector<double>& input, const std::vector<double>& target) {
    std::lock_guard<std::mutex> lock(Mutex_);
    std::vector<double> si = input;
    for (auto& v : si) v = std::clamp(v, -10.0, 10.0);
    // Forward
    std::vector<double> hid(HiddenSize_);
    for (size_t j = 0; j < HiddenSize_; ++j) {
        hid[j] = B1_[j];
        for (size_t i = 0; i < InputSize_; ++i) hid[j] += si[i] * W1_[i][j];
        hid[j] = std::max(0.0, hid[j]);
    }
    std::vector<double> out(OutputSize_);
    for (size_t k = 0; k < OutputSize_; ++k) {
        out[k] = B2_[k];
        for (size_t j = 0; j < HiddenSize_; ++j) out[k] += hid[j] * W2_[j][k];
        out[k] = std::clamp(out[k], -1.0, 1.0);
    }
    // Loss
    double loss = 0.0;
    for (size_t k = 0; k < OutputSize_; ++k) {
        double e = target[k] - out[k]; loss += e * e;
    }
    loss /= OutputSize_;
    // Output deltas
    std::vector<double> od(OutputSize_);
    for (size_t k = 0; k < OutputSize_; ++k)
        od[k] = std::clamp((target[k] - out[k]) * 2.0 / static_cast<double>(OutputSize_), -1.0, 1.0);
    // Update W2, B2
    for (size_t j = 0; j < HiddenSize_; ++j)
        for (size_t k = 0; k < OutputSize_; ++k)
            W2_[j][k] += std::clamp(LearningRate_ * hid[j] * od[k], -0.1, 0.1);
    for (size_t k = 0; k < OutputSize_; ++k)
        B2_[k] += std::clamp(LearningRate_ * od[k], -0.1, 0.1);
    // Hidden deltas
    std::vector<double> hd(HiddenSize_, 0.0);
    for (size_t j = 0; j < HiddenSize_; ++j) {
        for (size_t k = 0; k < OutputSize_; ++k) hd[j] += od[k] * W2_[j][k];
        hd[j] *= (hid[j] > 0.0) ? 1.0 : 0.0;
        hd[j] = std::clamp(hd[j], -1.0, 1.0);
    }
    // Update W1, B1
    for (size_t i = 0; i < InputSize_; ++i)
        for (size_t j = 0; j < HiddenSize_; ++j)
            W1_[i][j] += std::clamp(LearningRate_ * si[i] * hd[j], -0.1, 0.1);
    for (size_t j = 0; j < HiddenSize_; ++j)
        B1_[j] += std::clamp(LearningRate_ * hd[j], -0.1, 0.1);
    return loss;
}

void NeuralNetwork::SaveModel(const std::string& fn) {
    std::lock_guard<std::mutex> lock(Mutex_);
    std::ofstream f(fn);
    if (!f) throw std::runtime_error("Cannot save model: " + fn);
    f << InputSize_ << " " << HiddenSize_ << " " << OutputSize_ << " " << LearningRate_ << "\n";
    for (size_t i = 0; i < InputSize_; ++i) { for (size_t j = 0; j < HiddenSize_; ++j) f << W1_[i][j] << " "; f << "\n"; }
    for (size_t i = 0; i < HiddenSize_; ++i) { for (size_t j = 0; j < OutputSize_; ++j) f << W2_[i][j] << " "; f << "\n"; }
    for (size_t i = 0; i < HiddenSize_; ++i) f << B1_[i] << " "; f << "\n";
    for (size_t i = 0; i < OutputSize_; ++i) f << B2_[i] << " "; f << "\n";
}

void NeuralNetwork::LoadModel(const std::string& fn) {
    std::lock_guard<std::mutex> lock(Mutex_);
    std::ifstream f(fn);
    if (!f) throw std::runtime_error("Cannot load model: " + fn);
    f >> InputSize_ >> HiddenSize_ >> OutputSize_ >> LearningRate_;
    W1_.resize(InputSize_, std::vector<double>(HiddenSize_));
    W2_.resize(HiddenSize_, std::vector<double>(OutputSize_));
    B1_.resize(HiddenSize_); B2_.resize(OutputSize_); Hidden_.resize(HiddenSize_);
    for (size_t i = 0; i < InputSize_; ++i) for (size_t j = 0; j < HiddenSize_; ++j) f >> W1_[i][j];
    for (size_t i = 0; i < HiddenSize_; ++i) for (size_t j = 0; j < OutputSize_; ++j) f >> W2_[i][j];
    for (size_t i = 0; i < HiddenSize_; ++i) f >> B1_[i];
    for (size_t i = 0; i < OutputSize_; ++i) f >> B2_[i];
}

// ======================== SimpleProcessEnvironment ========================

SimpleProcessEnvironment::SimpleProcessEnvironment(double tc, double g, double dt, double dsd)
    : TimeConstant_(tc), Gain_(g), Dt_(dt), DisturbanceStdDev_(dsd),
      ProcessValue_(0.0), Setpoint_(0.0), StepCount_(0), MaxSteps_(1000),
      Gen_(std::random_device{}()), Dist_(0.0, dsd) {}

std::vector<double> SimpleProcessEnvironment::Reset() {
    StepCount_ = 0; ProcessValue_ = 0.0; Setpoint_ = 1.0;
    return {Setpoint_ - ProcessValue_, 0.0, 0.0};
}

std::tuple<std::vector<double>, double, bool>
SimpleProcessEnvironment::Step(const std::vector<double>&, double controlSignal) {
    double safeCtrl = std::clamp(controlSignal, -10.0, 10.0);
    double prevValue = ProcessValue_;
    double deriv = std::clamp((Gain_ * safeCtrl - ProcessValue_) / TimeConstant_, -1.0, 1.0);
    ProcessValue_ += Dt_ * deriv;
    if (DisturbanceStdDev_ > 0.0)
        ProcessValue_ += std::clamp(Dist_(Gen_), -0.1, 0.1);
    double prevError = Setpoint_ - prevValue;
    double error = Setpoint_ - ProcessValue_;
    double errorDeriv = std::clamp((error - prevError) / Dt_, -5.0, 5.0);
    double reward = -std::min(5.0, error * error) - 0.01 * std::min(5.0, safeCtrl * safeCtrl);
    if (std::abs(error) < 0.1) reward += 0.5;
    ++StepCount_;
    return {{error, errorDeriv, safeCtrl}, reward, StepCount_ >= MaxSteps_};
}

double SimpleProcessEnvironment::GetSetpoint() const { return Setpoint_; }
double SimpleProcessEnvironment::GetProcessValue() const { return ProcessValue_; }
void SimpleProcessEnvironment::SetSetpoint(double sp) { Setpoint_ = std::clamp(sp, -5.0, 5.0); }
void SimpleProcessEnvironment::SetMaxSteps(int ms) { MaxSteps_ = ms; }

// ======================== RlAgent ========================

RlAgent::RlAgent(size_t ss, size_t as, size_t hs, double lr, double g,
                  double eps, double ed, double em)
    : StateSize_(ss), ActionSize_(as),
      QNetwork_(ss, hs, as, lr), TargetNetwork_(ss, hs, as, lr),
      Gamma_(g), Epsilon_(eps), EpsilonDecay_(ed), EpsilonMin_(em),
      Gen_(std::random_device{}()),
      ActionRanges_({{0.0, 2.0}, {0.0, 1.0}, {0.0, 0.5}}) {
    SyncTargetNetwork();
}

std::vector<double> RlAgent::SelectAction(const std::vector<double>& state) {
    std::vector<double> action(ActionSize_);
    std::uniform_real_distribution<double> epsDist(0.0, 1.0);
    if (epsDist(Gen_) < Epsilon_) {
        for (size_t i = 0; i < ActionSize_; ++i) {
            std::uniform_real_distribution<double> d(ActionRanges_[i].first, ActionRanges_[i].second);
            action[i] = d(Gen_);
        }
    } else {
        auto qv = QNetwork_.Forward(state);
        for (size_t i = 0; i < ActionSize_; ++i) {
            double nq = (qv[i] + 1.0) / 2.0;
            double range = ActionRanges_[i].second - ActionRanges_[i].first;
            action[i] = ActionRanges_[i].first + nq * range;
        }
    }
    return action;
}

double RlAgent::Learn(const std::vector<double>& state, const std::vector<double>&,
                       double reward, const std::vector<double>& nextState, bool done) {
    auto nextQ = TargetNetwork_.Forward(nextState);
    auto qv = QNetwork_.Forward(state);
    std::vector<double> target = qv;
    for (size_t i = 0; i < ActionSize_; ++i) {
        target[i] = done ? std::clamp(reward, -1.0, 1.0)
                         : std::clamp(reward + Gamma_ * nextQ[i], -1.0, 1.0);
    }
    double loss = QNetwork_.Train(state, target);
    Epsilon_ = std::max(EpsilonMin_, Epsilon_ * EpsilonDecay_);
    return loss;
}

void RlAgent::SyncTargetNetwork() {
    std::string tmp = "/tmp/rl_sync_tmp.dat";
    QNetwork_.SaveModel(tmp);
    TargetNetwork_.LoadModel(tmp);
    std::remove(tmp.c_str());
}

void RlAgent::SaveModel(const std::string& fn) { QNetwork_.SaveModel(fn); }
void RlAgent::LoadModel(const std::string& fn) {
    QNetwork_.LoadModel(fn); TargetNetwork_.LoadModel(fn);
}
double RlAgent::GetEpsilon() const { return Epsilon_; }

std::vector<double> RlAgent::GridSearch(std::shared_ptr<Environment> env, int numTests) {
    std::vector<double> kpVals = {0.2, 0.5, 1.0, 1.5};
    std::vector<double> kiVals = {0.0, 0.1, 0.2, 0.5};
    std::vector<double> kdVals = {0.0, 0.02, 0.05, 0.1};
    double bestReward = -std::numeric_limits<double>::infinity();
    std::vector<double> bestParams = {0.5, 0.1, 0.05};
    PidController pid;
    for (double kp : kpVals) for (double ki : kiVals) for (double kd : kdVals) {
        pid.UpdateParameters(kp, ki, kd);
        env->Reset(); pid.Reset();
        double totalReward = 0.0;
        for (int s = 0; s < numTests; ++s) {
            double err = env->GetSetpoint() - env->GetProcessValue();
            double ctrl = pid.Compute(err);
            auto [st, r, done] = env->Step({kp, ki, kd}, ctrl);
            totalReward += r;
            if (done) break;
        }
        if (totalReward > bestReward) { bestReward = totalReward; bestParams = {kp, ki, kd}; }
    }
    return bestParams;
}

// ======================== SelfTuningPidController ========================

SelfTuningPidController::SelfTuningPidController(std::shared_ptr<Environment> env,
                                                   std::shared_ptr<PidController> pid,
                                                   size_t hs)
    : Env_(env), Pid_(pid), Agent_(3, 3, hs), IsTraining_(false) {}

void SelfTuningPidController::Train(int numEpisodes, int targetSyncFreq, bool verbose) {
    IsTraining_.store(true);
    double bestReward = -std::numeric_limits<double>::infinity();
    std::vector<double> bestParams;
    for (int ep = 0; ep < numEpisodes && IsTraining_.load(); ++ep) {
        auto state = Env_->Reset();
        Pid_->Reset();
        double epReward = 0.0;
        bool done = false;
        int steps = 0;
        while (!done && steps < 200 && IsTraining_.load()) {
            auto action = Agent_.SelectAction(state);
            Pid_->UpdateParameters(action[0], action[1], action[2]);
            double err = Env_->GetSetpoint() - Env_->GetProcessValue();
            double ctrl = Pid_->Compute(err);
            auto [nextState, reward, d] = Env_->Step(action, ctrl);
            done = d;
            if (steps > 0) Agent_.Learn(state, action, reward, nextState, done);
            state = nextState;
            epReward += reward;
            ++steps;
        }
        if (ep % targetSyncFreq == 0) Agent_.SyncTargetNetwork();
        if (epReward > bestReward) {
            bestReward = epReward;
            bestParams = Pid_->GetParameters();
        }
        if (verbose)
            std::cout << "Episode " << ep + 1 << "/" << numEpisodes
                      << "  reward=" << std::fixed << std::setprecision(2) << epReward
                      << "  eps=" << Agent_.GetEpsilon() << "\n";
    }
    if (!bestParams.empty())
        Pid_->UpdateParameters(bestParams[0], bestParams[1], bestParams[2]);
    IsTraining_.store(false);
}

void SelfTuningPidController::SimpleTuning(int numTests) {
    auto best = Agent_.GridSearch(Env_, numTests);
    Pid_->UpdateParameters(best[0], best[1], best[2]);
}

std::vector<std::vector<double>> SelfTuningPidController::Run(int numSteps, double setpoint) {
    std::vector<std::vector<double>> history;
    Env_->Reset(); Pid_->Reset();
    if (auto* pe = dynamic_cast<SimpleProcessEnvironment*>(Env_.get()))
        pe->SetSetpoint(setpoint);
    auto params = Pid_->GetParameters();
    for (int s = 0; s < numSteps; ++s) {
        double t = s * 0.01;
        double err = Env_->GetSetpoint() - Env_->GetProcessValue();
        double ctrl = Pid_->Compute(err);
        Env_->Step(params, ctrl);
        history.push_back({t, Env_->GetSetpoint(), Env_->GetProcessValue(), ctrl});
    }
    return history;
}

void SelfTuningPidController::SaveModel(const std::string& fn) { Agent_.SaveModel(fn); }
void SelfTuningPidController::LoadModel(const std::string& fn) { Agent_.LoadModel(fn); }
void SelfTuningPidController::StopTraining() { IsTraining_.store(false); }

} // namespace Rl
