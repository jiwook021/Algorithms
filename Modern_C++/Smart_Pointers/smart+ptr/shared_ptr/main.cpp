#include <iostream>
#include <utility> // for std::swap

template<typename T>
class SharedPtr {
private:
    T* ptr;         // 관리하는 원시 포인터
    int* ref_count; // 참조 카운터

    // 내부에서 ref_count를 감소시키고, 0이 되면 메모리를 해제하는 함수
    void release() {
        if (ref_count) {
            --(*ref_count);
            if (*ref_count == 0) {
                delete ptr;
                delete ref_count;
            }
        }
        ptr = nullptr;
        ref_count = nullptr;
    }

public:
    // 생성자: raw 포인터를 받아 소유권을 가짐
    explicit SharedPtr(T* p = nullptr) : ptr(p) {
        if (ptr) {
            ref_count = new int(1);
        } else {
            ref_count = nullptr;
        }
    }

    // 복사 생성자: 참조 카운터를 증가시킴
    SharedPtr(const SharedPtr& other) : ptr(other.ptr), ref_count(other.ref_count) {
        if (ref_count) {
            ++(*ref_count);
        }
    }

    // 이동 생성자: 다른 SharedPtr에서 소유권을 가져옴
    SharedPtr(SharedPtr&& other) noexcept : ptr(other.ptr), ref_count(other.ref_count) {
        other.ptr = nullptr;
        other.ref_count = nullptr;
    }

    // 소멸자: 참조 카운터를 감소시키고, 0이면 리소스를 해제함
    ~SharedPtr() {
        release();
    }

    // 복사 대입 연산자
    SharedPtr& operator=(const SharedPtr& other) {
        if (this != &other) {
            release();  // 기존 리소스 해제
            ptr = other.ptr;
            ref_count = other.ref_count;
            if (ref_count) {
                ++(*ref_count);
            }
        }
        return *this;
    }

    // 이동 대입 연산자
    SharedPtr& operator=(SharedPtr&& other) noexcept {
        if (this != &other) {
            release();  // 기존 리소스 해제
            ptr = other.ptr;
            ref_count = other.ref_count;
            other.ptr = nullptr;
            other.ref_count = nullptr;
        }
        return *this;
    }

    // 역참조 연산자
    T& operator*() const { return *ptr; }

    // 멤버 접근 연산자
    T* operator->() const { return ptr; }

    // 내부 포인터 반환
    T* get() const { return ptr; }

    // 현재 참조 카운트 반환
    int use_count() const { return ref_count ? *ref_count : 0; }

    // 소유한 객체를 해제하고, 새 포인터로 교체
    void reset(T* p = nullptr) {
        release();
        if (p) {
            ptr = p;
            ref_count = new int(1);
        }
    }

    // 다른 SharedPtr와 내부 포인터 및 참조 카운터를 교환
    void swap(SharedPtr& other) {
        std::swap(ptr, other.ptr);
        std::swap(ref_count, other.ref_count);
    }
};

// 테스트용 클래스
struct Test {
    int value;
    Test(int v) : value(v) {
        std::cout << "Test(" << value << ") constructed\n";
    }
    ~Test() {
        std::cout << "Test(" << value << ") destroyed\n";
    }
    void hello() const {
        std::cout << "Hello from Test(" << value << ")\n";
    }
};

int main() {
    {
        // sp1이 Test 객체의 소유권을 가짐
        SharedPtr<Test> sp1(new Test(10));
        std::cout << "sp1 use_count: " << sp1.use_count() << "\n";

        {
            // sp2가 sp1을 복사함으로써, 동일 객체를 공유
            SharedPtr<Test> sp2 = sp1;
            std::cout << "After copy, sp1 use_count: " << sp1.use_count() << "\n";
            sp2->hello();
        } // sp2 소멸 → 참조 카운트 감소

        std::cout << "After inner block, sp1 use_count: " << sp1.use_count() << "\n";
    } // sp1 소멸 → 참조 카운트 0이 되어 Test 객체와 ref_count 해제

    return 0;
}
