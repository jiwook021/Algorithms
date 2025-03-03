#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <functional>

namespace custom_api {

// NeuralNetwork 클래스: torch::nn::Sequential을 랩핑한 간단한 모델 API
struct NeuralNetworkImpl : torch::nn::Module {
    torch::nn::Sequential layers;

    NeuralNetworkImpl() {
        layers = torch::nn::Sequential();
        register_module("layers", layers);
    }

    // 템플릿을 이용하여 임의의 모듈을 추가합니다.
    template <typename ModuleType>
    void addLayer(ModuleType layer) {
        layers->push_back(layer);
    }

    // 순전파 함수
    torch::Tensor forward(torch::Tensor x) {
        return layers->forward(x);
    }
};
TORCH_MODULE(NeuralNetwork);  // NeuralNetworkImpl에 대한 스마트 포인터 정의

// Trainer 클래스: 모델, 옵티마이저, 손실함수를 캡슐화하여 학습 루프 제공
class Trainer {
public:
    NeuralNetwork model;
    std::unique_ptr<torch::optim::Optimizer> optimizer;
    // 손실 함수: 두 텐서를 받아 손실 값을 계산하는 람다 또는 함수 포인터
    std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> loss_fn;

    Trainer(NeuralNetwork model,
            std::unique_ptr<torch::optim::Optimizer> optimizer,
            std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> loss_fn)
        : model(model), optimizer(std::move(optimizer)), loss_fn(loss_fn) {}

    // 간단한 fit 함수: 주어진 데이터와 타겟으로 여러 에포크 학습
    void fit(torch::Tensor data, torch::Tensor targets, int epochs) {
        model->train();
        for (int epoch = 0; epoch < epochs; ++epoch) {
            optimizer->zero_grad();            // 기울기 초기화
            auto output = model->forward(data);  // 순전파 실행
            auto loss = loss_fn(output, targets); // 손실 계산
            loss.backward();                     // 역전파 실행
            optimizer->step();                   // 파라미터 업데이트

            std::cout << "Epoch " << epoch 
                      << " | Loss: " << loss.item<double>() << std::endl;
        }
    }
};

} // namespace custom_api

// main 함수: 사용자 API를 이용하여 모델 구성 및 학습 실행
int main() {
    // 모델 생성 (입력은 784 차원 벡터로 가정)
    auto model = custom_api::NeuralNetwork();

    // Sequential 방식으로 레이어를 추가합니다.
    // 첫 번째 레이어: Linear (784 -> 64)
    model->addLayer(torch::nn::Linear(784, 64));
    // ReLU 활성화 함수 적용
    model->addLayer(torch::nn::Functional(torch::relu));
    // 두 번째 레이어: Linear (64 -> 10)
    model->addLayer(torch::nn::Linear(64, 10));
    // 출력에 log_softmax 적용 (dim=1)
    model->addLayer(torch::nn::Functional(
        [](const torch::Tensor &input) {
            return torch::log_softmax(input, /*dim=*/1);
        }
    ));

    // 옵티마이저 생성: SGD 사용 (학습률 0.01)
    auto optimizer = std::make_unique<torch::optim::SGD>(
        model->parameters(), torch::optim::SGDOptions(0.01)
    );

    // 손실 함수 정의: Negative Log-Likelihood Loss
    auto loss_fn = [](const torch::Tensor &output, const torch::Tensor &target) {
        return torch::nll_loss(output, target);
    };

    // Trainer 객체 생성: 모델, 옵티마이저, 손실 함수를 전달합니다.
    custom_api::Trainer trainer(model, std::move(optimizer), loss_fn);

    // 학습용 더미 데이터 생성:
    // 배치 크기 64, 입력 벡터 크기 784
    auto input = torch::randn({64, 784});
    // 타겟: 0~9 사이의 임의의 정수 (클래스 10개)
    auto target = torch::randint(0, 10, {64});

    // 5 에포크 동안 학습
    trainer.fit(input, target, 5);

    return 0;
}
