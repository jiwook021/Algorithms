#include <iostream>
#include <functional>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

// 1. 기본 함수 - 인자가 3개인 함수
int add(int a, int b, int c) {
    std::cout << "add(" << a << ", " << b << ", " << c << ") 호출됨" << std::endl;
    return a + b + c;
}

// 2. 다양한 타입의 인자를 받는 함수
std::string makeGreeting(const std::string& name, int age, bool formal) {
    if (formal) {
        return name + "님, 만 " + std::to_string(age) + "세이시군요.";
    } else {
        return "안녕 " + name + "! 너는 " + std::to_string(age) + "살이구나!";
    }
}

// 3. 멤버 함수를 가진 클래스
class Calculator {
public:
    int multiply(int x, int y) const {
        std::cout << "Calculator::multiply(" << x << ", " << y << ") 호출됨" << std::endl;
        return x * y;
    }
    
    double divide(double x, double y) const {
        if (y == 0) throw std::runtime_error("0으로 나눌 수 없습니다");
        std::cout << "Calculator::divide(" << x << ", " << y << ") 호출됨" << std::endl;
        return x / y;
    }
    
    void printResult(const std::string& operation, double result) const {
        std::cout << operation << " 결과: " << result << std::endl;
    }
};

// 4. 함수 객체(Functor)
class Adder {
private:
    int base;
    
public:
    Adder(int base) : base(base) {}
    
    int operator()(int x) const {
        std::cout << "Adder(" << base << ")(" << x << ") 호출됨" << std::endl;
        return base + x;
    }
};

int main() {
    using namespace std::placeholders;  // _1, _2, _3 등의 플레이스홀더를 사용하기 위함
    
    std::cout << "=== 1. 기본 함수 바인딩 예제 ===" << std::endl;
    
    // 1.1 첫 번째 인자를 10으로 고정
    auto add10 = std::bind(add, 10, _1, _2);
    std::cout << "add10(20, 30) = " << add10(20, 30) << std::endl;  // add(10, 20, 30)
    
    // 1.2 첫 번째와 세 번째 인자를 고정
    auto add5and15 = std::bind(add, 5, _1, 15);
    std::cout << "add5and15(10) = " << add5and15(10) << std::endl;  // add(5, 10, 15)
    
    // 1.3 모든 인자를 고정
    auto addFixed = std::bind(add, 1, 2, 3);
    std::cout << "addFixed() = " << addFixed() << std::endl;  // add(1, 2, 3)
    
    // 1.4 인자 순서 바꾸기
    auto addReordered = std::bind(add, _3, _1, _2);
    std::cout << "addReordered(10, 20, 30) = " << addReordered(10, 20, 30) << std::endl;  // add(30, 10, 20)
    
    std::cout << "\n=== 2. 복잡한 함수 바인딩 예제 ===" << std::endl;
    
    // 2.1 formal 인자를 true로 고정
    auto formalGreeting = std::bind(makeGreeting, _1, _2, true);
    std::cout << formalGreeting("김철수", 30) << std::endl;
    
    // 2.2 모든 인자를 다른 순서로 재배치
    auto reorderedGreeting = std::bind(makeGreeting, _2, _3, _1);
    std::cout << reorderedGreeting(false, "박영희", 25) << std::endl;  // false가 formal, "박영희"가 name, 25가 age로 전달됨
    
    std::cout << "\n=== 3. 클래스 멤버 함수 바인딩 예제 ===" << std::endl;
    
    Calculator calc;
    
    // 3.1 객체의 멤버 함수 바인딩
    auto mult = std::bind(&Calculator::multiply, &calc, _1, _2);
    std::cout << "mult(6, 7) = " << mult(6, 7) << std::endl;
    
    // 3.2 첫 번째 인자를 고정한 멤버 함수
    auto multiplyBy5 = std::bind(&Calculator::multiply, &calc, 5, _1);
    std::cout << "multiplyBy5(6) = " << multiplyBy5(6) << std::endl;
    
    // 3.3 참조로 객체 전달
    std::shared_ptr<Calculator> calcPtr = std::make_shared<Calculator>();
    auto dividePtr = std::bind(&Calculator::divide, calcPtr, _1, _2);
    std::cout << "dividePtr(10, 2) = " << dividePtr(10, 2) << std::endl;
    
    // 3.4 객체와 결과 출력 함수 연결
    auto printMultiply = std::bind(&Calculator::printResult, &calc, "곱셈", 
                                  std::bind(&Calculator::multiply, &calc, _1, _2));
    printMultiply(10, 20);  // "곱셈 결과: 200" 출력
    
    std::cout << "\n=== 4. 함수 객체(Functor) 바인딩 예제 ===" << std::endl;
    
    // 4.1 함수 객체 생성 및 바인딩
    Adder add20(20);
    auto boundAdder = std::bind(add20, _1);
    std::cout << "boundAdder(15) = " << boundAdder(15) << std::endl;
    
    // 4.2 함수 객체를 인자로 직접 전달
    auto directBoundAdder = std::bind(Adder(25), _1);
    std::cout << "directBoundAdder(15) = " << directBoundAdder(15) << std::endl;
    
    std::cout << "\n=== 5. 알고리즘과 함께 사용하는 예제 ===" << std::endl;
    
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // 5.1 특정 값보다 큰 숫자 찾기
    int threshold = 6;
    auto count = std::count_if(numbers.begin(), numbers.end(), 
                              std::bind(std::greater<int>(), _1, threshold));
    std::cout << threshold << "보다 큰 숫자의 개수: " << count << std::endl;
    
    // 5.2 각 요소에 특정 값 곱하기
    std::transform(numbers.begin(), numbers.end(), numbers.begin(),
                  std::bind(std::multiplies<int>(), _1, 2));
    
    std::cout << "각 요소에 2를 곱한 결과: ";
    for (int num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\n=== 6. 람다와 비교 예제 ===" << std::endl;
    
    // bind 사용
    auto bindAdd5 = std::bind(add, 5, _1, _2);
    std::cout << "bindAdd5(10, 15) = " << bindAdd5(10, 15) << std::endl;
    
    // 동일한 기능을 하는 람다
    auto lambdaAdd5 = [](int b, int c) { return add(5, b, c); };
    std::cout << "lambdaAdd5(10, 15) = " << lambdaAdd5(10, 15) << std::endl;
    
    return 0;
}