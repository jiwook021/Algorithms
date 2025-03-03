#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

class ProducerConsumerQueue {
private:
    std::queue<int> queue;               // 데이터 저장 큐
    std::mutex mtx;                      // 뮤텍스
    std::condition_variable cv_full;     // 큐가 가득 찼을 때 대기
    std::condition_variable cv_empty;    // 큐가 비었을 때 대기
    const size_t max_size = 7;           // 큐의 최대 크기

public:
    // 프로듀서: 큐에 데이터 추가
    void produce(int value) {
        std::unique_lock<std::mutex> lock(mtx);
        // 큐가 가득 찼으면 대기
        cv_full.wait(lock, [this] { return queue.size() < max_size; });

        queue.push(value);
        std::cout << "Produced: " << value << " (Queue size: " << queue.size() << ")\n";

        lock.unlock();
        cv_empty.notify_one(); // 컨슈머에게 데이터가 추가됨을 알림
    }

    // 컨슈머: 큐에서 데이터 꺼내기
    int consume() {
        std::unique_lock<std::mutex> lock(mtx);
        // 큐가 비었으면 대기
        cv_empty.wait(lock, [this] { return !queue.empty(); });

        int value = queue.front();
        queue.pop();
        std::cout << "Consumed: " << value << " (Queue size: " << queue.size() << ")\n";

        lock.unlock();
        cv_full.notify_one(); // 프로듀서에게 공간이 생겼음을 알림
        return value;
    }
};

// 프로듀서 스레드 함수
void producer_task(ProducerConsumerQueue& pcq, int id) {
    for (int i = 0; i < 5; ++i) {
        int value = id * 10 + i; // 고유 값 생성 (예: ID 1 -> 10, 11, 12...)
        pcq.produce(value);
        std::this_thread::sleep_for(std::chrono::milliseconds(5000)); // 생산 속도 조절
    }
}

// 컨슈머 스레드 함수
void consumer_task(ProducerConsumerQueue& pcq, int id) {
    for (int i = 0; i < 5; ++i) {
        pcq.consume();
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 소비 속도 조절
    }
}

int main() {
    ProducerConsumerQueue pcq;

    // 두 개의 프로듀서와 두 개의 컨슈머 스레드 생성
    std::thread producers[2];
    std::thread consumers[2];

    for (int i = 0; i < 2; ++i) {
        producers[i] = std::thread(producer_task, std::ref(pcq), i + 1);
        consumers[i] = std::thread(consumer_task, std::ref(pcq), i + 1);
    }

    // 모든 스레드 종료 대기
    for (int i = 0; i < 2; ++i) {
        producers[i].join();
        consumers[i].join();
    }

    std::cout << "모든 작업 완료\n";
    return 0;
}