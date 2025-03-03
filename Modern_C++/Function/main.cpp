#include <iostream>
#include <functional>

// 1. 일반 함수
void normalFunction() {
    std::cout << "일반 함수 호출됨" << std::endl;
}

// 2. 함수 객체 클래스
class FunctionObject {
public:
    void operator()() const {
        std::cout << "함수 객체 호출됨" << std::endl;
    }
};

int main() {
    // std::function 객체 선언
    std::function<void()> task;
    
    // 1. 일반 함수 저장
    task = normalFunction;
    task();  // 출력: 일반 함수 호출됨
    
    // 2. 함수 객체 저장
    FunctionObject functor;
    task = functor;
    task();  // 출력: 함수 객체 호출됨
    
    // 3. 람다 함수 저장
    task = []() {
        std::cout << "람다 함수 호출됨" << std::endl;
    };
    task();  // 출력: 람다 함수 호출됨
    
    // 4. bind를 사용한 멤버 함수와 객체 바인딩
    class MyClass {
    public:
        void memberFunction() {
            std::cout << "멤버 함수 호출됨" << std::endl;
        }
    };
    
    MyClass obj;
    task = std::bind(&MyClass::memberFunction, &obj);
    task();  // 출력: 멤버 함수 호출됨
    
    return 0;
}