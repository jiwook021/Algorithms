#include <iostream>
#include <vector>
#include <functional>
#include <stdexcept>
#include <algorithm>

template <typename T, typename Comparator = std::less<T>>
class BinaryHeap {
private:
    std::vector<T> data;
    Comparator comp;

    // Bubble up the element at index to its correct position.
    void bubbleUp(size_t index) {
        while (index > 0) {
            size_t parent = (index - 1) / 2;
            // If the current element has higher priority than its parent, swap.
            if (comp(data[index], data[parent])) {
                std::swap(data[index], data[parent]);
                index = parent;
            } else {
                break;
            }
        }
    }

    // Bubble down the element at index to restore heap property.
    void bubbleDown(size_t index) {
        size_t size = data.size();
        while (true) {
            size_t left = 2 * index + 1;
            size_t right = 2 * index + 2;
            size_t best = index;  // best is the index with highest priority.

            if (left < size && comp(data[left], data[best])) {
                best = left;
            }
            if (right < size && comp(data[right], data[best])) {
                best = right;
            }
            if (best != index) {
                std::swap(data[index], data[best]);
                index = best;
            } else {
                break;
            }
        }
    }

public:
    BinaryHeap() : data(), comp(Comparator()) {}

    // Inserts a new element into the heap.
    void push(const T& value) {
        data.push_back(value);
        bubbleUp(data.size() - 1);
    }

    // Removes the top element from the heap.
    void pop() {
        if (empty()) {
            throw std::out_of_range("Heap is empty");
        }
        // Move the last element to the top and then bubble it down.
        data[0] = data.back();
        data.pop_back();
        if (!empty()) {
            bubbleDown(0);
        }
    }

    // Returns the top element.
    const T& top() const {
        if (empty()) {
            throw std::out_of_range("Heap is empty");
        }
        return data[0];
    }

    // Checks if the heap is empty.
    bool empty() const {
        return data.empty();
    }

    // Returns the number of elements in the heap.
    size_t size() const {
        return data.size();
    }
};

int main() {
    // Example: Create a min-heap of integers.
    BinaryHeap<int, std::less<int>> minHeap;
    minHeap.push(5);
    minHeap.push(3);
    minHeap.push(8);
    minHeap.push(1);

    std::cout << "Min-Heap (smallest on top): ";
    while (!minHeap.empty()) {
        std::cout << minHeap.top() << " ";
        minHeap.pop();
    }
    std::cout << "\n";

    // // Example: Create a max-heap of integers.
    // BinaryHeap<int, std::greater<int>> maxHeap;
    // maxHeap.push(5);
    // maxHeap.push(3);
    // maxHeap.push(8);
    // maxHeap.push(1);

    // std::cout << "Max-Heap (largest on top): ";
    // while (!maxHeap.empty()) {
    //     std::cout << maxHeap.top() << " ";
    //     maxHeap.pop();
    // }
    // std::cout << "\n";

    return 0;
}











#define QUEUESIZE 10 
class Queue 
{
    Queue(int size)
    {
        head = 0; 
        rear = size; 
    }

    void push(int data)
    {
        queue[rear] = data; 
        rear = (rear+1)%QUEUESIZE ; 
    }

    int pop()
    {
        int retval = queue[head];
        head = (head +1)%QUEUESIZE; 
        return retval;   
    }
    
    private: 
    int size; 
    int queue[QUEUESIZE]; 
    int head;
    int rear; 


}

int main()
{
    
    Queue queue(0,10); 
    queue.push(5);
    queue.push(3);
    queue.push(7);
    

}


#include <stdio.h>
#include <stdint.h>

// To execute C, please define "int main()"


// Extract BITS from a integer(int val, int lsb, int width)

// b0000 0000 1010 1111, lsb = 1, width = 4 : b0111
// b0000 0000 1010 1111, lsb = 1, width = 4 : 0000 1110
int extract_bits(const int val, int lsb, int width) {
 if(lsb <0 || width <0|| lsb+width>sizeof(int)*8)
    return -1;
  int mask = ((1<<width) -1) <<lsb;
  return mask & val; 
}

int main()
{
  int val = 0x00AF;
  int lsb =1;
  int width = 4; 
  printf("0x%X",extract_bits(val, lsb,width)); 
}
/* int main() {
  // Write a C program to find out if the underlying architecture is little endian or big endian

  uint16_t number = 0xabcd; 
  uint8_t *byteptr = (uint8_t *) &number; 
  if(byteptr[0] == 0xcd)  
    printf("little\n");
  else
    printf("big\n");
} */


// Your previous Plain Text content is preserved below:


// #include <stdio.h>
// int main ()
// {
//   // Write a C program to find out if the underlying architecture is little endian or big endian

//   uint8_t number = 0x01; 
//   uint16_t *byteptr = (uint8_t *) &number; 
//   if(byteptr[0] == 0x01)  
//     printf("little\n");
//   else
//     printf("big\n");
// }







// class A {
//   virtual void fun1(int x) = 0;
// }

// class B : A
// {
//   void fun1(int x) {return};
// }

// B objB;
// objB.fun1() // virtual not there?? classA::func1
// // virtual is there?? classB::func1

// vptr->func1()




    const int val = 10; //값 const 
    int* mutable_ptr = const_cast<int*>(&val);
    // *mutable_ptr = 20; // 주의: Undefined Behavior (val은 원래 const)
    std::cout << "Value: " << val << std::endl;

    // 2. 실제 수정 가능한 경우
    int original = 5;
    const int* const_ptr = &original;
    int* non_const_ptr = const_cast<int*>(const_ptr);
    *non_const_ptr = 15; // 안전: original은 const가 아님
    std::cout << "Modified: " << original << std::endl; // 출력: 15

    // 3. 함수 호출에서 const 제거
    int data = 10;
    const int* const_data = &data;
    modifyValue(const_cast<int*>(const_data));
    std::cout << "After function: " << data << std::endl; // 출력: 20

    return 0;

    const int a; 
    const int *b; // b의값이 상수인거고  
    const int* const c;// c의 값도 상수 c 의 pointer도 상수 
const int* const c; 

int* a = &b. 

a-> b의 주소 



#include <cstdio>

//c라는 친구는 결국 a 의 값 복사본  

void function(int* c) {
    int new_value = 20; 
    c = &new_value; // c만 변경, a에 영향 없음
}

int main() {
    int b = 10;
    int* a = &b; //a = b의주소를;  (a*)메모리주소가있음 
    printf("Before: %d\n", *a); // 10
    function(a); //b의주소를 가지고있는 친구 
    printf("After: %d\n", *a); // 10 (변경 안 됨)
    return 0;
}





#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

class ProducerConsumerQueue {
private:
    std::queue<int> queue;              // 데이터 저장 큐
    std::mutex mtx;                     // 뮤텍스
    std::condition_variable cv_producer; // 프로듀서용 조건 변수
    std::condition_variable cv_consumer; // 컨슈머용 조건 변수
    const size_t max_size = 5;          // 큐의 최대 크기

public:
    // 프로듀서: 데이터를 큐에 추가
    void produce(int value) {
        std::unique_lock<std::mutex> lock(mtx);
        // 큐가 가득 찼으면 대기
        cv_producer.wait(lock, [this] { return queue.size() < max_size; });
        queue.push(value);
        std::cout << "Produced: " << value << std::endl;
        lock.unlock();
        cv_consumer.notify_one(); // 컨슈머에게 데이터가 추가되었음을 알림
    }

    // 컨슈머: 큐에서 데이터를 꺼내 처리
    void consume() {
        std::unique_lock<std::mutex> lock(mtx);
        // 큐가 비었으면 대기
        cv_consumer.wait(lock, [this] { return !queue.empty(); });
        int value = queue.front();
        queue.pop();
        std::cout << "Consumed: " << value << std::endl;
        lock.unlock();
        cv_producer.notify_one(); // 프로듀서에게 큐에 공간이 생겼음을 알림
    }
};

// 프로듀서 스레드 함수
void producer_task(ProducerConsumerQueue& pcq) {
    for (int i = 0; i < 10; ++i) {
        pcq.produce(i);
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 생산 속도 조절
    }
}

// 컨슈머 스레드 함수
void consumer_task(ProducerConsumerQueue& pcq) {
    for (int i = 0; i < 10; ++i) {
        pcq.consume();
        std::this_thread::sleep_for(std::chrono::milliseconds(200)); // 소비 속도 조절
    }
}

int main() {
    ProducerConsumerQueue pcq;

    // 스레드 생성
    std::thread producer(producer_task, std::ref(pcq));
    std::thread consumer(consumer_task, std::ref(pcq));

    // 스레드 종료 대기
    producer.join();
    consumer.join();

    return 0;
}




#include <iostream>
#include <thread>
#include <mutex>

std::mutex mtx;
int counter = 0;

void incrementCounter(int id, int times) {
    for (int i = 0; i < times; ++i) {
        std::lock_guard<std::mutex> lock(mtx); // 자동으로 잠금/해제
        counter++;
        std::cout << "스레드 " << id << ": " << counter << "\n";
    }
}

int main() {
    std::thread t1(incrementCounter, 1, 5);
    std::thread t2(incrementCounter, 2, 5);

    t1.join();
    t2.join();

    std::cout << "최종 counter 값: " << counter << "\n";
    return 0;
}