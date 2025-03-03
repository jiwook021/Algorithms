#include <iostream>
#include <utility> // std::swap

// 간단한 unique_ptr 구현
template <typename T>
class UniquePtr {
private:
    T* ptr; // 관리하는 원시 포인터
public:
    // 기본 생성자: 빈 unique_ptr
    UniquePtr() : ptr(nullptr) {}

    // raw 포인터를 받아 소유권을 가져감
    explicit UniquePtr(T* p) : ptr(p) {}

    // 소멸자: 소유한 객체를 삭제함
    ~UniquePtr() {
        delete ptr;
    }

    // 복사 생성자 및 복사 대입 연산자는 삭제 (unique_ptr은 유일 소유)
    UniquePtr(const UniquePtr&) = delete;
    UniquePtr& operator=(const UniquePtr&) = delete;

    // 이동 생성자: 다른 unique_ptr의 소유권을 가져옴
    UniquePtr(UniquePtr&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    // 이동 대입 연산자: 다른 unique_ptr의 소유권을 가져오고, 기존 객체는 삭제
    UniquePtr& operator=(UniquePtr&& other) noexcept {
        if (this != &other) {
            delete ptr;        // 기존 소유 객체 삭제
            ptr = other.ptr;   // 소유권 이전
            other.ptr = nullptr;
        }
        return *this;
    }

    // 역참조 연산자: 소유한 객체에 접근
    T& operator*() const {
        return *ptr;
    }

    // 멤버 접근 연산자: 소유한 객체의 멤버에 접근
    T* operator->() const {
        return ptr;
    }

    // 내부 포인터 반환
    T* get() const {
        return ptr;
    }

    // 소유권 포기: 내부 포인터를 반환하고, unique_ptr는 nullptr로 설정
    T* release() {
        T* temp = ptr;
        ptr = nullptr;
        return temp;
    }

    // 소유 객체를 교체: 기존 객체 삭제 후 새로운 포인터 소유
    void reset(T* p = nullptr) {
        if (ptr != p) {
            delete ptr;
            ptr = p;
        }
    }

    // 두 unique_ptr의 내부 포인터를 교환
    void swap(UniquePtr& other) noexcept {
        std::swap(ptr, other.ptr);
    }
};

// 테스트용 클래스
struct Test {
    int value;
    Test(int v) : value(v) { std::cout << "Test(" << value << ") constructed\n"; }
    ~Test() { std::cout << "Test(" << value << ") destroyed\n"; }
    void hello() const { std::cout << "Hello from Test(" << value << ")\n"; }
};

int main() {
    {
        UniquePtr<Test> ptr1(new Test(10));
        ptr1->hello();
        
        // // 이동 생성자 테스트
         UniquePtr<Test> ptr2 = std::move(ptr1);
        // if (!ptr1.get()) {
        //     std::cout << "ptr1 is now null after move\n";
        // }
        // ptr2->hello();

        // 이동 대입 연산자 테스트
        UniquePtr<Test> ptr3;
        ptr3 = std::move(ptr2);
        if (!ptr2.get()) {
            std::cout << "ptr2 is now null after move assignment\n";
        }
        ptr3->hello();

        // reset 테스트: 기존 객체 삭제 후 새 객체 소유
        ptr3.reset(new Test(20));
        ptr3->hello();
        
        // release 테스트: 소유권 포기 후, 수동 delete 필요
        Test* raw = ptr3.release();
        if (!ptr3.get()) {
            std::cout << "ptr3 is null after release\n";
        }
        delete raw;
    } // 블록 종료 시 남은 unique_ptr는 자동으로 소유한 객체를 삭제
    return 0;
}
