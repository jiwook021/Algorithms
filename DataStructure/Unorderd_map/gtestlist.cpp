#include <iostream>
#include <vector>
#include <functional>
#include <utility>
#include <stdexcept>
#include <iterator>
#include <limits>
#include <algorithm>
#include <memory>
#include <gtest/gtest.h>
#include <tuple> // for std::piecewise_construct, std::forward_as_tuple

// -----------------------------------------------------------------------------
// List 클래스 템플릿 구현
// 이중 연결 리스트로, STL의 std::list와 유사한 인터페이스를 제공
// 더미 노드를 사용하여 빈 리스트와 경계 조건 처리를 단순화
// -----------------------------------------------------------------------------
template <typename T>
class List {
private:
    // 리스트 노드 구조체 - 데이터와 이전/다음 노드 포인터를 가짐
    struct Node {
        T data;               // 노드가 저장하는 데이터
        Node* prev;           // 이전 노드에 대한 포인터
        Node* next;           // 다음 노드에 대한 포인터

        // 기본 생성자 (더미 노드용) - 데이터 초기화 없이 포인터만 nullptr로 설정
        Node() : prev(nullptr), next(nullptr) {}

        // 값을 받는 생성자 - 복사 생성
        explicit Node(const T& value) : data(value), prev(nullptr), next(nullptr) {}
        
        // 이동 생성자 - 이동 의미론 지원
        explicit Node(T&& value) : data(std::move(value)), prev(nullptr), next(nullptr) {}
        
        // 가변 인자 템플릿으로 in-place 생성 - 인자를 완벽하게 전달하여 데이터를 제자리 생성
        template<typename... Args>
        explicit Node(Args&&... args) : data(std::forward<Args>(args)...), prev(nullptr), next(nullptr) {}
    };

    // 리스트의 더미 노드들 (시작 전과 끝 후)
    Node* head;               // 더미 노드 (시작 전) - 첫 번째 실제 노드 이전
    Node* tail;               // 더미 노드 (끝 후) - 마지막 실제 노드 이후
    size_t node_count;        // 리스트에 있는 노드 수 (더미 노드 제외)

public:
    // 반복자 클래스 정의 - 양방향 반복자(bidirectional iterator) 구현
    class iterator {
    private:
        Node* node_ptr;      // 현재 노드를 가리키는 포인터
        friend class List;   // List 클래스가 private 멤버에 접근할 수 있도록 함

    public:
        // 반복자 특성 타입 정의 - STL 호환성을 위함
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        // 기본 생성자 - nullptr로 초기화
        iterator() : node_ptr(nullptr) {}

        // 노드 포인터를 받는 생성자
        explicit iterator(Node* node) : node_ptr(node) {}

        // 역참조 연산자 - 노드의 데이터를 반환
        T& operator*() const {
            return node_ptr->data;
        }

        // 화살표 연산자 - 노드 데이터의 멤버에 접근할 수 있게 함
        T* operator->() const {
            return &(node_ptr->data);
        }

        // 전위 증가 연산자 - 다음 노드로 이동 후 자신을 반환
        iterator& operator++() {
            if (node_ptr)
                node_ptr = node_ptr->next;
            return *this;
        }

        // 후위 증가 연산자 - 현재 위치의 복사본을 반환하고 다음 노드로 이동
        iterator operator++(int) {
            iterator temp = *this;
            ++(*this);
            return temp;
        }

        // 전위 감소 연산자 - 이전 노드로 이동 후 자신을 반환
        iterator& operator--() {
            if (node_ptr)
                node_ptr = node_ptr->prev;
            return *this;
        }

        // 후위 감소 연산자 - 현재 위치의 복사본을 반환하고 이전 노드로 이동
        iterator operator--(int) {
            iterator temp = *this;
            --(*this);
            return temp;
        }

        // 비교 연산자 - 두 반복자가 같은 노드를 가리키는지 확인
        bool operator==(const iterator& other) const {
            return node_ptr == other.node_ptr;
        }

        // 비교 연산자 - 두 반복자가 다른 노드를 가리키는지 확인
        bool operator!=(const iterator& other) const {
            return node_ptr != other.node_ptr;
        }
    };

    // const_iterator 클래스 정의 - 읽기 전용 반복자
    class const_iterator {
    private:
        const Node* node_ptr;  // 현재 노드를 가리키는 상수 포인터
        friend class List;     // List 클래스가 private 멤버에 접근할 수 있도록 함

    public:
        // 반복자 특성 타입 정의 - STL 호환성을 위함
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        // 기본 생성자 - nullptr로 초기화
        const_iterator() : node_ptr(nullptr) {}

        // 노드 포인터를 받는 생성자
        explicit const_iterator(const Node* node) : node_ptr(node) {}
        
        // iterator로부터 변환 생성자 - 일반 반복자를 상수 반복자로 변환
        const_iterator(const iterator& it) : node_ptr(it.node_ptr) {}

        // 역참조 연산자 - 노드의 데이터를 상수 참조로 반환
        const T& operator*() const {
            return node_ptr->data;
        }

        // 화살표 연산자 - 노드 데이터의 멤버에 상수 접근 제공
        const T* operator->() const {
            return &(node_ptr->data);
        }

        // 전위 증가 연산자 - 다음 노드로 이동 후 자신을 반환
        const_iterator& operator++() {
            if (node_ptr)
                node_ptr = node_ptr->next;
            return *this;
        }

        // 후위 증가 연산자 - 현재 위치의 복사본을 반환하고 다음 노드로 이동
        const_iterator operator++(int) {
            const_iterator temp = *this;
            ++(*this);
            return temp;
        }

        // 전위 감소 연산자 - 이전 노드로 이동 후 자신을 반환
        const_iterator& operator--() {
            if (node_ptr)
                node_ptr = node_ptr->prev;
            return *this;
        }

        // 후위 감소 연산자 - 현재 위치의 복사본을 반환하고 이전 노드로 이동
        const_iterator operator--(int) {
            const_iterator temp = *this;
            --(*this);
            return temp;
        }

        // 비교 연산자 - 두 상수 반복자가 같은 노드를 가리키는지 확인
        bool operator==(const const_iterator& other) const {
            return node_ptr == other.node_ptr;
        }

        // 비교 연산자 - 두 상수 반복자가 다른 노드를 가리키는지 확인
        bool operator!=(const const_iterator& other) const {
            return node_ptr != other.node_ptr;
        }
    };

    // 생성자 - 빈 리스트 초기화, 더미 노드 생성
    List() : node_count(0) {
        head = new Node(); // 시작 전 더미 노드
        tail = new Node(); // 끝 후 더미 노드
        
        // 초기 상태: head <-> tail (빈 리스트)
        head->next = tail;
        tail->prev = head;
    }

    // 복사 생성자 - 다른 리스트의 모든 요소를 복사
    List(const List& other) : node_count(0) {
        // 더미 노드 초기화
        head = new Node();
        tail = new Node();
        head->next = tail;
        tail->prev = head;

        // 다른 리스트의 모든 요소를 복사
        for (auto it = other.begin(); it != other.end(); ++it) {
            push_back(*it);
        }
    }

    // 이동 생성자 - 다른 리스트의 내용을 이동(훔치기)
    List(List&& other) noexcept : head(other.head), tail(other.tail), node_count(other.node_count) {
        // 다른 리스트의 상태를 리셋하여 소멸자에서 문제가 생기지 않도록 함
        other.head = nullptr;
        other.tail = nullptr;
        other.node_count = 0;
    }

    // 복사 할당 연산자 - 다른 리스트의 모든 요소를 복사
    List& operator=(const List& other) {
        if (this != &other) {  // 자기 할당 검사
            clear();  // 기존 요소 모두 제거
            
            // 다른 리스트의 모든 요소를 복사
            for (auto it = other.begin(); it != other.end(); ++it) {
                push_back(*it);
            }
        }
        return *this;
    }

    // 이동 할당 연산자 - 다른 리스트의 내용을 이동(훔치기)
    List& operator=(List&& other) noexcept {
        if (this != &other) {  // 자기 할당 검사
            // 기존 리소스 정리
            clear();
            delete head;
            delete tail;

            // 다른 리스트의 내용을 가져옴
            head = other.head;
            tail = other.tail;
            node_count = other.node_count;

            // 다른 리스트의 상태를 리셋
            other.head = nullptr;
            other.tail = nullptr;
            other.node_count = 0;
        }
        return *this;
    }

    // 소멸자 - 모든 노드 삭제 및 메모리 해제
    ~List() {
        clear();  // 모든 요소 삭제
        delete head;  // 더미 헤드 노드 삭제
        delete tail;  // 더미 테일 노드 삭제
    }

    // 반복자 접근 메서드 - 첫 번째 요소를 가리키는 반복자 반환
    iterator begin() {
        return iterator(head->next);
    }

    // 상수 반복자 접근 메서드 - 첫 번째 요소를 가리키는 상수 반복자 반환
    const_iterator begin() const {
        return const_iterator(head->next);
    }

    // 상수 반복자 접근 메서드 - 첫 번째 요소를 가리키는 상수 반복자 반환 (명시적 const 버전)
    const_iterator cbegin() const {
        return const_iterator(head->next);
    }

    // 반복자 접근 메서드 - 마지막 요소 다음을 가리키는 반복자 반환
    iterator end() {
        return iterator(tail);
    }

    // 상수 반복자 접근 메서드 - 마지막 요소 다음을 가리키는 상수 반복자 반환
    const_iterator end() const {
        return const_iterator(tail);
    }

    // 상수 반복자 접근 메서드 - 마지막 요소 다음을 가리키는 상수 반복자 반환 (명시적 const 버전)
    const_iterator cend() const {
        return const_iterator(tail);
    }

    // 크기 관련 메서드 - 리스트가 비어 있는지 확인
    bool empty() const {
        return node_count == 0;
    }

    // 크기 관련 메서드 - 리스트의 요소 수 반환
    size_t size() const {
        return node_count;
    }

    // 요소 접근 메서드 - 첫 번째 요소 참조 반환
    T& front() {
        if (empty()) throw std::out_of_range("List is empty");
        return head->next->data;
    }

    // 요소 접근 메서드 - 첫 번째 요소 상수 참조 반환
    const T& front() const {
        if (empty()) throw std::out_of_range("List is empty");
        return head->next->data;
    }

    // 요소 접근 메서드 - 마지막 요소 참조 반환
    T& back() {
        if (empty()) throw std::out_of_range("List is empty");
        return tail->prev->data;
    }

    // 요소 접근 메서드 - 마지막 요소 상수 참조 반환
    const T& back() const {
        if (empty()) throw std::out_of_range("List is empty");
        return tail->prev->data;
    }

    // 요소 추가 메서드 - 값을 리스트 끝에 추가
    void push_back(const T& value) {
        Node* new_node = new Node(value);  // 새 노드 생성
        insert_before(tail, new_node);     // tail 더미 노드 앞에 삽입
    }

    // 요소 추가 메서드 - 이동 의미론을 사용하여 값을 리스트 끝에 추가
    void push_back(T&& value) {
        Node* new_node = new Node(std::move(value));  // 새 노드 생성 (이동)
        insert_before(tail, new_node);                // tail 더미 노드 앞에 삽입
    }

    // emplace_back: 가변 인자 템플릿을 사용하여 요소를 리스트 끝에 제자리 생성
    template<typename... Args>
    T& emplace_back(Args&&... args) {
        Node* new_node = new Node(std::forward<Args>(args)...);  // 인수를 완벽하게 전달하여 노드 생성
        insert_before(tail, new_node);                          // tail 더미 노드 앞에 삽입
        return tail->prev->data;                                // 새로 생성된 요소 참조 반환
    }

    // 요소 제거 메서드 - 마지막 요소 삭제
    void pop_back() {
        if (empty()) throw std::out_of_range("List is empty");
        erase_node(tail->prev);  // 마지막 실제 노드 삭제 (tail 바로 앞)
    }

    // 요소 제거 메서드 - 첫 번째 요소 삭제
    void pop_front() {
        if (empty()) throw std::out_of_range("List is empty");
        erase_node(head->next);  // 첫 번째 실제 노드 삭제 (head 바로 뒤)
    }

    // 특정 위치의 요소 삭제 - 반복자가 가리키는 요소 삭제 후 다음 요소 반복자 반환
    iterator erase(iterator pos) {
        if (pos.node_ptr == head || pos.node_ptr == tail) {
            return end();  // 더미 노드는 삭제할 수 없음
        }
        
        Node* next_node = pos.node_ptr->next;  // 다음 노드 저장
        erase_node(pos.node_ptr);              // 현재 노드 삭제
        return iterator(next_node);            // 다음 노드 반복자 반환
    }

    // 모든 요소 삭제 - 리스트를 비움
    void clear() {
        Node* current = head->next;  // 첫 번째 실제 노드
        while (current != tail) {    // 모든 실제 노드를 순회
            Node* next = current->next;  // 다음 노드 저장
            delete current;              // 현재 노드 삭제
            current = next;              // 다음 노드로 이동
        }
        
        // 빈 리스트 상태로 초기화
        head->next = tail;
        tail->prev = head;
        node_count = 0;
    }

private:
    // 노드 삽입 헬퍼 함수 - 지정된 노드 앞에 새 노드 삽입
    void insert_before(Node* position, Node* new_node) {
        // position 앞에 new_node 삽입
        new_node->next = position;                // 새 노드의 다음을 position으로 설정
        new_node->prev = position->prev;          // 새 노드의 이전을 position의 이전으로 설정
        position->prev->next = new_node;          // position 이전 노드의 다음을 새 노드로 설정
        position->prev = new_node;                // position의 이전을 새 노드로 설정
        
        ++node_count;  // 노드 수 증가
    }

    // 노드 삭제 헬퍼 함수 - 지정된 노드 삭제
    void erase_node(Node* node) {
        // node의 prev와 next를 서로 연결하여 node를 리스트에서 제외
        node->prev->next = node->next;  // node 이전 노드의 다음을 node의 다음으로 설정
        node->next->prev = node->prev;  // node 다음 노드의 이전을 node의 이전으로 설정
        
        delete node;  // 노드 메모리 해제
        --node_count; // 노드 수 감소
    }
};

// -----------------------------------------------------------------------------
// UnorderedMap 클래스 템플릿 - 해시 테이블 기반 맵 구현
// Key         : 키 타입
// T           : 값 타입
// Hash        : 해시 함수 (기본: std::hash<Key>)
// KeyEqual    : 키 비교 함수 (기본: std::equal_to<Key>)
//
// ※ 내부에서는 버킷 관리를 위해 std::vector와 List를 사용함.
// -----------------------------------------------------------------------------
template <typename Key, typename T, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key> >
class UnorderedMap {
private:
    // 기본 버킷 수
    static const size_t default_bucket_count = 8;

    // 버킷은 List<std::pair<const Key, T> >로 구현 - 충돌 처리를 위한 체이닝 방식 사용
    std::vector< List< std::pair<const Key, T> > > buckets;

    // 저장된 전체 요소 수
    size_t num_elements;

    // 해시 함수와 키 비교 객체
    Hash hash_func;
    KeyEqual key_equal_obj;

    // 내부: 주어진 키의 버킷 인덱스 계산 - 해시값을 버킷 수로 모듈로 연산
    size_t bucket_index(const Key & key) const {
        return hash_func(key) % buckets.size();
    }

public:
    // ======================================================
    // 생성자 / 소멸자
    // ======================================================
    
    // 기본 생성자 - 기본 버킷 수로 맵 초기화
    UnorderedMap()
        : buckets(default_bucket_count),  // 기본 버킷 수로 벡터 초기화
          num_elements(0),                // 요소 수 0으로 초기화
          hash_func(Hash()),              // 기본 해시 함수 초기화
          key_equal_obj(KeyEqual())       // 기본 키 비교 함수 초기화
    {
    }

    // 소멸자 - 모든 요소 제거
    ~UnorderedMap() {
        clear();
    }

    // ======================================================
    // Capacity 관련 멤버 함수
    // ======================================================
    
    // 맵이 비어 있는지 확인
    bool empty() const {
        return num_elements == 0;
    }

    // 저장된 요소 수 반환
    size_t size() const {
        return num_elements;
    }

    // 최대 가능 요소 수 반환 (이론적 한계)
    size_t max_size() const {
        return std::numeric_limits<size_t>::max();
    }

    // ======================================================
    // Iterator 구현
    // ======================================================
    // iterator 클래스 (non-const) - 맵의 모든 요소를 순회할 수 있는 반복자
    class iterator {
    private:
        UnorderedMap * map_ptr;           // UnorderedMap 객체 포인터
        size_t bucket_idx;                // 현재 버킷 인덱스
        // 현재 버킷 내 반복자
        typename List< std::pair<const Key, T> >::iterator bucket_iter;

        // 빈 버킷을 건너뛰는 함수 - 유효한 요소를 가리키도록 반복자 조정
        void advance_to_valid() {
            while (map_ptr && bucket_idx < map_ptr->buckets.size() &&
                   bucket_iter == map_ptr->buckets[bucket_idx].end())
            {
                ++bucket_idx;  // 다음 버킷으로 이동
                if (bucket_idx < map_ptr->buckets.size())
                    bucket_iter = map_ptr->buckets[bucket_idx].begin();  // 다음 버킷의 첫 요소
            }
        }

        // UnorderedMap이 private 멤버에 접근할 수 있도록 friend 선언
        friend class UnorderedMap;
        // const_iterator와 비교를 위해 friend 선언
        friend class const_iterator;
    public:
        // 기본 생성자 - 빈 반복자 생성
        iterator() : map_ptr(nullptr), bucket_idx(0) {
        }

        // 맵, 버킷 인덱스, 버킷 내 반복자를 받는 생성자
        iterator(UnorderedMap * m, size_t idx, typename List< std::pair<const Key, T> >::iterator it)
            : map_ptr(m), bucket_idx(idx), bucket_iter(it)
        {
            if (map_ptr)
                advance_to_valid();  // 유효한 요소를 가리키도록 조정
        }

        // 역참조 연산자 - 현재 요소(키-값 쌍) 참조 반환
        std::pair<const Key, T> & operator*() const {
            return *bucket_iter;
        }

        // 화살표 연산자 - 현재 요소의 멤버에 접근
        std::pair<const Key, T> * operator->() const {
            return &(*bucket_iter);
        }

        // 전위 증가 연산자 - 다음 요소로 이동
        iterator & operator++() {
            ++bucket_iter;        // 버킷 내 다음 요소로 이동
            advance_to_valid();   // 유효한 요소를 가리키도록 조정
            return *this;
        }

        // 후위 증가 연산자 - 다음 요소로 이동하고 이전 위치 반환
        iterator operator++(int) {
            iterator temp = *this;
            ++(*this);
            return temp;
        }

        // 비교 연산자 - 두 반복자가 같은 요소를 가리키는지 확인
        // 수정된 operator==: 만약 두 반복자 모두 bucket_idx가 buckets.size()라면 end 상태로 간주함
        bool operator==(const iterator & other) const {
            if (map_ptr != other.map_ptr)
                return false;
            if (bucket_idx == map_ptr->buckets.size() && other.bucket_idx == map_ptr->buckets.size())
                return true;  // 두 반복자 모두 맵의 끝을 가리킴
            return bucket_idx == other.bucket_idx && bucket_iter == other.bucket_iter;
        }

        // 비교 연산자 - 두 반복자가 다른 요소를 가리키는지 확인
        bool operator!=(const iterator & other) const {
            return !(*this == other);
        }
    };

    // const_iterator 클래스 (읽기 전용) - 맵의 모든 요소를 순회할 수 있는 읽기 전용 반복자
    class const_iterator {
    private:
        const UnorderedMap * map_ptr;  // UnorderedMap 객체 포인터 (const)
        size_t bucket_idx;             // 현재 버킷 인덱스
        // 현재 버킷 내 const 반복자
        typename List< std::pair<const Key, T> >::const_iterator bucket_iter;

        // 빈 버킷을 건너뛰는 함수 - 유효한 요소를 가리키도록 반복자 조정
        void advance_to_valid() {
            while (map_ptr && bucket_idx < map_ptr->buckets.size() &&
                   bucket_iter == map_ptr->buckets[bucket_idx].end())
            {
                ++bucket_idx;  // 다음 버킷으로 이동
                if (bucket_idx < map_ptr->buckets.size())
                    bucket_iter = map_ptr->buckets[bucket_idx].begin();  // 다음 버킷의 첫 요소
            }
        }

        friend class UnorderedMap;
    public:
        // 기본 생성자 - 빈 상수 반복자 생성
        const_iterator() : map_ptr(nullptr), bucket_idx(0) {
        }

        // 맵, 버킷 인덱스, 버킷 내 상수 반복자를 받는 생성자
        const_iterator(const UnorderedMap * m, size_t idx, typename List< std::pair<const Key, T> >::const_iterator it)
            : map_ptr(m), bucket_idx(idx), bucket_iter(it)
        {
            if (map_ptr)
                advance_to_valid();  // 유효한 요소를 가리키도록 조정
        }

        // 역참조 연산자 - 현재 요소(키-값 쌍)의 상수 참조 반환
        const std::pair<const Key, T> & operator*() const {
            return *bucket_iter;
        }

        // 화살표 연산자 - 현재 요소의 멤버에 상수 접근
        const std::pair<const Key, T> * operator->() const {
            return &(*bucket_iter);
        }

        // 전위 증가 연산자 - 다음 요소로 이동
        const_iterator & operator++() {
            ++bucket_iter;        // 버킷 내 다음 요소로 이동
            advance_to_valid();   // 유효한 요소를 가리키도록 조정
            return *this;
        }

        // 후위 증가 연산자 - 다음 요소로 이동하고 이전 위치 반환
        const_iterator operator++(int) {
            const_iterator temp = *this;
            ++(*this);
            return temp;
        }

        // 비교 연산자 - 두 상수 반복자가 같은 요소를 가리키는지 확인
        bool operator==(const const_iterator & other) const {
            if (map_ptr != other.map_ptr)
                return false;
            if (bucket_idx == map_ptr->buckets.size() && other.bucket_idx == map_ptr->buckets.size())
                return true;  // 두 반복자 모두 맵의 끝을 가리킴
            return bucket_idx == other.bucket_idx && bucket_iter == other.bucket_iter;
        }

        // 비교 연산자 - 두 상수 반복자가 다른 요소를 가리키는지 확인
        bool operator!=(const const_iterator & other) const {
            return !(*this == other);
        }
    };

    // ======================================================
    // Iterator 접근자
    // ======================================================
    
    // begin - 첫 번째 요소를 가리키는 반복자 반환
    iterator begin() {
        // 첫 번째 비어 있지 않은 버킷을 찾음
        for (size_t i = 0; i < buckets.size(); ++i) {
            if (!buckets[i].empty())
                return iterator(this, i, buckets[i].begin());
        }
        return end();  // 모든 버킷이 비어 있으면 end() 반환
    }

    // end - 마지막 요소 다음을 가리키는 반복자 반환
    iterator end() {
        // end()는 bucket_idx가 buckets.size()인 상태로 생성 (마지막 버킷 이후)
        return iterator(this, buckets.size(), typename List< std::pair<const Key, T> >::iterator());
    }

    // begin - 첫 번째 요소를 가리키는 상수 반복자 반환 (const 버전)
    const_iterator begin() const {
        // 첫 번째 비어 있지 않은 버킷을 찾음
        for (size_t i = 0; i < buckets.size(); ++i) {
            if (!buckets[i].empty())
                return const_iterator(this, i, buckets[i].begin());
        }
        return end();  // 모든 버킷이 비어 있으면 end() 반환
    }

    // end - 마지막 요소 다음을 가리키는 상수 반복자 반환 (const 버전)
    const_iterator end() const {
        return const_iterator(this, buckets.size(), typename List< std::pair<const Key, T> >::const_iterator());
    }

    // cbegin - 첫 번째 요소를 가리키는 상수 반복자 반환
    const_iterator cbegin() const {
        return begin();
    }

    // cend - 마지막 요소 다음을 가리키는 상수 반복자 반환
    const_iterator cend() const {
        return end();
    }

    // ======================================================
    // Element Access
    // ======================================================
    // operator[] - 키에 해당하는 값 참조 반환, 키가 없으면 기본값으로 새 요소 생성
    // 수정: piecewise_construct 사용하여 in-place 생성
    T & operator[](const Key & key) {
        iterator it = find(key);
        if (it == end()) {
            // 키가 없으면 새 요소 생성
            size_t idx = bucket_index(key);
            
            // std::piecewise_construct를 사용하여 요소 in-place 생성
            auto& inserted = buckets[idx].emplace_back(
                std::piecewise_construct,
                std::forward_as_tuple(key),     // key는 const Key&로 전달
                std::forward_as_tuple()         // T()의 인자 없음 (기본 생성)
            );
            
            ++num_elements;  // 요소 수 증가
            return inserted.second;  // 새로 삽입된 값 참조 반환
        }
        return it->second;  // 기존 값 참조 반환
    }

    // operator[] - 이동 의미론을 사용한 버전, 키가 없으면 기본값으로 새 요소 생성
    // 수정: rvalue 참조 지원 추가
    T & operator[](Key && key) {
        iterator it = find(key);
        if (it == end()) {
            // 키가 없으면 새 요소 생성
            size_t idx = bucket_index(key);
            
            // std::piecewise_construct를 사용하여 요소 in-place 생성
            auto& inserted = buckets[idx].emplace_back(
                std::piecewise_construct,
                std::forward_as_tuple(std::move(key)),  // key는 Key&&로 이동
                std::forward_as_tuple()                 // T()의 인자 없음 (기본 생성)
            );
            
            ++num_elements;  // 요소 수 증가
            return inserted.second;  // 새로 삽입된 값 참조 반환
        }
        return it->second;  // 기존 값 참조 반환
    }

    // at - 키에 해당하는 값 참조 반환, 키가 없으면 예외 발생
    T & at(const Key & key) {
        iterator it = find(key);
        if (it == end())
            throw std::out_of_range("Key not found");  // 키를 찾지 못하면 예외 발생
        return it->second;  // 값 참조 반환
    }

    // at - 키에 해당하는 값 상수 참조 반환, 키가 없으면 예외 발생 (const 버전)
    const T & at(const Key & key) const {
        const_iterator it = find(key);
        if (it == end())
            throw std::out_of_range("Key not found");  // 키를 찾지 못하면 예외 발생
        return it->second;  // 값 상수 참조 반환
    }

    // ======================================================
    // Lookup
    // ======================================================
    // find - 키에 해당하는 요소를 찾아 반복자 반환
    iterator find(const Key & key) {
        size_t idx = bucket_index(key);  // 키의 해시값으로 버킷 인덱스 계산
        typename List< std::pair<const Key, T> >::iterator it = buckets[idx].begin();
        for (; it != buckets[idx].end(); ++it) {
            if (key_equal_obj(it->first, key))  // 키 비교
                return iterator(this, idx, it);  // 찾으면 해당 요소의 반복자 반환
        }
        return end();  // 찾지 못하면 end() 반환
    }

    // find - 키에 해당하는 요소를 찾아 상수 반복자 반환 (const 버전)
    const_iterator find(const Key & key) const {
        size_t idx = bucket_index(key);  // 키의 해시값으로 버킷 인덱스 계산
        typename List< std::pair<const Key, T> >::const_iterator it = buckets[idx].begin();
        for (; it != buckets[idx].end(); ++it) {
            if (key_equal_obj(it->first, key))  // 키 비교
                return const_iterator(this, idx, it);  // 찾으면 해당 요소의 상수 반복자 반환
        }
        return end();  // 찾지 못하면 end() 반환
    }

    // count - 키에 해당하는 요소의 수 반환 (0 또는 1)
    size_t count(const Key & key) const {
        return (find(key) == end()) ? 0 : 1;  // 키가 있으면 1, 없으면 0 반환
    }

    // equal_range - 키에 해당하는 요소 범위 반환
    std::pair<iterator, iterator> equal_range(const Key & key) {
        iterator it = find(key);
        if (it == end())
            return std::pair<iterator, iterator>(end(), end());  // 키가 없으면 빈 범위 반환
        iterator next = it;
        ++next;  // 다음 요소로 이동
        return std::pair<iterator, iterator>(it, next);  // 키가 있으면 [it, next) 범위 반환
    }

    // equal_range - 키에 해당하는 요소 범위 반환 (const 버전)
    std::pair<const_iterator, const_iterator> equal_range(const Key & key) const {
        const_iterator it = find(key);
        if (it == end())
            return std::pair<const_iterator, const_iterator>(end(), end());  // 키가 없으면 빈 범위 반환
        const_iterator next = it;
        ++next;  // 다음 요소로 이동
        return std::pair<const_iterator, const_iterator>(it, next);  // 키가 있으면 [it, next) 범위 반환
    }

    // ======================================================
    // Modifiers
    // ======================================================
    // emplace - 전달받은 인자를 이용해 새 요소 생성 후 삽입
    template <typename... Args>
    std::pair<iterator, bool> emplace(Args&&... args) {
        // 먼저 임시 pair 객체 생성
        auto temp_pair = std::pair<const Key, T>(std::forward<Args>(args)...);
        const Key& key = temp_pair.first;
        
        // 이미 존재하는지 확인
        size_t idx = bucket_index(key);
        typename List< std::pair<const Key, T> >::iterator it = buckets[idx].begin();
        for (; it != buckets[idx].end(); ++it) {
            if (key_equal_obj(it->first, key))
                return std::pair<iterator, bool>(iterator(this, idx, it), false);  // 키가 있으면 삽입하지 않고 반환
        }
        
        // 요소 in-place 생성 (이동)
        buckets[idx].push_back(std::move(temp_pair));
        ++num_elements;  // 요소 수 증가
        it = buckets[idx].end();
        --it;  // 새로 삽입된 요소의 반복자 얻기
        return std::pair<iterator, bool>(iterator(this, idx, it), true);  // 새 요소 삽입 성공
    }

    // insert - lvalue 기반 삽입
    std::pair<iterator, bool> insert(const std::pair<const Key, T> & val) {
        size_t idx = bucket_index(val.first);  // 키의 해시값으로 버킷 인덱스 계산
        typename List< std::pair<const Key, T> >::iterator it = buckets[idx].begin();
        for (; it != buckets[idx].end(); ++it) {
            if (key_equal_obj(it->first, val.first))
                return std::pair<iterator, bool>(iterator(this, idx, it), false);  // 키가 있으면 삽입하지 않고 반환
        }
        buckets[idx].push_back(val);  // 키가 없으면 요소 삽입
        ++num_elements;  // 요소 수 증가
        it = buckets[idx].end();
        --it;  // 새로 삽입된 요소의 반복자 얻기
        return std::pair<iterator, bool>(iterator(this, idx, it), true);  // 새 요소 삽입 성공
    }

    // insert - rvalue 기반 삽입 (이동 의미론)
    std::pair<iterator, bool> insert(std::pair<const Key, T> && val) {
        size_t idx = bucket_index(val.first);  // 키의 해시값으로 버킷 인덱스 계산
        typename List< std::pair<const Key, T> >::iterator it = buckets[idx].begin();
        for (; it != buckets[idx].end(); ++it) {
            if (key_equal_obj(it->first, val.first))
                return std::pair<iterator, bool>(iterator(this, idx, it), false);  // 키가 있으면 삽입하지 않고 반환
        }
        buckets[idx].push_back(std::move(val));  // 키가 없으면 요소 이동 삽입
        ++num_elements;  // 요소 수 증가
        it = buckets[idx].end();
        --it;  // 새로 삽입된 요소의 반복자 얻기
        return std::pair<iterator, bool>(iterator(this, idx, it), true);  // 새 요소 삽입 성공
    }

    // erase - 키 기반 삭제, 존재하면 1, 아니면 0 반환
    size_t erase(const Key & key) {
        size_t idx = bucket_index(key);  // 키의 해시값으로 버킷 인덱스 계산
        typename List< std::pair<const Key, T> >::iterator it = buckets[idx].begin();
        for (; it != buckets[idx].end(); ++it) {
            if (key_equal_obj(it->first, key)) {  // 키 비교
                buckets[idx].erase(it);  // 요소 삭제
                --num_elements;  // 요소 수 감소
                return 1;  // 삭제 성공
            }
        }
        return 0;  // 키를 찾지 못함
    }

    // erase - 반복자 기반 삭제, 다음 요소의 반복자 반환
    iterator erase(iterator pos) {
        if (pos == end())
            return pos;  // 끝 반복자는 삭제할 요소가 없음
        size_t idx = pos.bucket_idx;  // 버킷 인덱스
        auto next_it = buckets[idx].erase(pos.bucket_iter);  // 요소 삭제 후 다음 요소 반복자 받기
        --num_elements;  // 요소 수 감소
        return iterator(this, idx, next_it);  // 다음 요소 반복자 반환
    }

    // clear - 모든 요소 삭제
    void clear() {
        for (size_t i = 0; i < buckets.size(); ++i) {
            buckets[i].clear();  // 각 버킷의 모든 요소 삭제
        }
        num_elements = 0;  // 요소 수 초기화
    }

    // swap - 다른 UnorderedMap과 내용 교환
    void swap(UnorderedMap & other) {
        std::swap(buckets, other.buckets);  // 버킷 벡터 교환
        std::swap(num_elements, other.num_elements);  // 요소 수 교환
        std::swap(hash_func, other.hash_func);  // 해시 함수 교환
        std::swap(key_equal_obj, other.key_equal_obj);  // 키 비교 함수 교환
    }

    // ======================================================
    // Bucket 인터페이스
    // ======================================================
    // bucket_count - 현재 버킷 수 반환
    size_t bucket_count() const {
        return buckets.size();
    }

    // max_bucket_count - 최대 가능 버킷 수 반환 (이론적 한계)
    size_t max_bucket_count() const {
        return std::numeric_limits<size_t>::max();
    }

    // bucket_size - 지정된 버킷의 요소 수 반환
    size_t bucket_size(size_t n) const {
        return (n < buckets.size() ? buckets[n].size() : 0);  // 유효한 버킷 인덱스인지 확인
    }

    // bucket - 키가 속한 버킷 인덱스 반환
    size_t bucket(const Key & key) const {
        return bucket_index(key);  // 키의 해시값으로 버킷 인덱스 계산
    }

    // ======================================================
    // Hash 및 키 비교 객체 접근자
    // ======================================================
    // hash_function - 사용 중인 해시 함수 반환
    Hash hash_function() const {
        return hash_func;
    }

    // key_eq - 사용 중인 키 비교 함수 반환
    KeyEqual key_eq() const {
        return key_equal_obj;
    }
};

//-----------------------------------------------------------------------------
// Google Test 테스트 케이스 - UnorderedMap의 기능을 검증하는 단위 테스트
//-----------------------------------------------------------------------------

// 기본 테스트 클래스 설정 - 모든 테스트의 기본 환경 준비
class UnorderedMapTest : public ::testing::Test {
protected:
    // 테스트 전에 실행되는 메서드 - 테스트 환경 설정
    void SetUp() override {
        // 테스트 전 설정 코드
    }

    // 테스트 후에 실행되는 메서드 - 테스트 환경 정리
    void TearDown() override {
        // 테스트 후 정리 코드
    }

    // 맵에 테스트 데이터를 채우는 헬퍼 함수 - 여러 테스트에서 공통으로 사용
    void populateMap(UnorderedMap<int, std::string>& map) {
        map.insert(std::pair<const int, std::string>(1, "one"));
        map.insert(std::pair<const int, std::string>(2, "two"));
        map[3] = "three";
        map[4] = "four";
        map[5] = "five";
    }
};

// 생성자 및 기본 용량 메서드 테스트 - 맵의 기본 상태 확인
TEST_F(UnorderedMapTest, ConstructorAndCapacity) {
    UnorderedMap<int, std::string> umap;
    EXPECT_TRUE(umap.empty());  // 새로 생성된 맵은 비어 있어야 함
    EXPECT_EQ(0, umap.size());  // 새로 생성된 맵의 크기는 0이어야 함
    
    // max_size는 큰 값이어야 함
    EXPECT_GT(umap.max_size(), 0);
    
    // 초기 버킷 수는 기본값과 일치해야 함
    EXPECT_EQ(8, umap.bucket_count());
}

// 삽입 및 요소 접근 테스트 - 요소 추가 및 접근 기능 확인
TEST_F(UnorderedMapTest, InsertionAndAccess) {
    UnorderedMap<int, std::string> umap;
    
    // insert 테스트
    auto result = umap.insert(std::pair<const int, std::string>(1, "one"));
    EXPECT_TRUE(result.second);  // 성공적으로 삽입되어야 함
    EXPECT_EQ("one", result.first->second);
    EXPECT_EQ(1, umap.size());
    
    // 중복 키 insert 테스트
    result = umap.insert(std::pair<const int, std::string>(1, "another_one"));
    EXPECT_FALSE(result.second);  // 삽입되지 않아야 함
    EXPECT_EQ("one", result.first->second);  // 값은 변경되지 않아야 함
    EXPECT_EQ(1, umap.size());
    
    // operator[] 테스트
    umap[2] = "two";
    EXPECT_EQ(2, umap.size());
    EXPECT_EQ("two", umap[2]);
    
    // 존재하는 키에 대한 at() 테스트
    EXPECT_EQ("one", umap.at(1));
    
    // 존재하지 않는 키에 대한 at() 테스트
    EXPECT_THROW(umap.at(3), std::out_of_range);
    
    // operator[]가 기본값을 생성하는지 테스트
    EXPECT_EQ("", umap[3]);  // 기본 초기화된 문자열
    EXPECT_EQ(3, umap.size());
    
    EXPECT_EQ("", umap[4]); 
    EXPECT_EQ(4, umap.size());
    
    // rvalue key 테스트
    int temp_key = 4;
    umap[std::move(temp_key)] = "four";
    EXPECT_EQ("four", umap[4]);
    EXPECT_EQ(4, umap.size());
}

// emplace 테스트 - 인자로부터 직접 요소 생성 기능 확인
TEST_F(UnorderedMapTest, Emplace) {
    UnorderedMap<int, std::string> umap;
    
    // 기본 emplace 테스트
    auto result = umap.emplace(1, "one");
    EXPECT_TRUE(result.second);
    EXPECT_EQ("one", result.first->second);
    
    // 중복 키 emplace 테스트
    result = umap.emplace(1, "another");
    EXPECT_FALSE(result.second);
    EXPECT_EQ("one", result.first->second);  // 값은 변경되지 않아야 함
    
    // 새로운 키 emplace 테스트
    result = umap.emplace(2, "two");
    EXPECT_TRUE(result.second);
    EXPECT_EQ("two", result.first->second);
    
    EXPECT_EQ(2, umap.size());
}

// find 및 count 테스트 - 요소 조회 기능 확인
TEST_F(UnorderedMapTest, LookupOperations) {
    UnorderedMap<int, std::string> umap;
    populateMap(umap);
    
    // 존재하는 키에 대한 find 테스트
    auto it = umap.find(3);
    EXPECT_NE(umap.end(), it);
    EXPECT_EQ("three", it->second);
    
    // 존재하지 않는 키에 대한 find 테스트
    it = umap.find(6);
    EXPECT_EQ(umap.end(), it);
    
    // 존재하는 키에 대한 count 테스트
    EXPECT_EQ(1, umap.count(4));
    
    // 존재하지 않는 키에 대한 count 테스트
    EXPECT_EQ(0, umap.count(6));
    
    // 존재하는 키에 대한 equal_range 테스트
    auto range = umap.equal_range(5);
    EXPECT_NE(range.first, umap.end());
    EXPECT_EQ("five", range.first->second);
    
    // 다음 항목으로 증가
    auto temp = range.first;
    ++temp;
    EXPECT_EQ(temp, range.second);
    
    // 존재하지 않는 키에 대한 equal_range 테스트
    range = umap.equal_range(6);
    EXPECT_EQ(range.first, umap.end());
    EXPECT_EQ(range.second, umap.end());
}

// 삭제 연산 테스트 - 요소 제거 기능 확인
TEST_F(UnorderedMapTest, EraseOperations) {
    UnorderedMap<int, std::string> umap;
    populateMap(umap);
    EXPECT_EQ(5, umap.size());
    
    // 키별 삭제 테스트
    size_t count = umap.erase(3);
    EXPECT_EQ(1, count);
    EXPECT_EQ(4, umap.size());
    EXPECT_EQ(0, umap.count(3));
    
    // 존재하지 않는 키 삭제 테스트
    count = umap.erase(6);
    EXPECT_EQ(0, count);
    EXPECT_EQ(4, umap.size());
    
    // 반복자별 삭제 테스트
    auto it = umap.find(2);
    auto next_it = umap.erase(it);
    EXPECT_EQ(3, umap.size());
    EXPECT_EQ(0, umap.count(2));
    EXPECT_NE(umap.end(), next_it);  // 유효한 요소를 가리켜야 함
    
    // clear 테스트
    umap.clear();
    EXPECT_TRUE(umap.empty());
    EXPECT_EQ(0, umap.size());
    EXPECT_EQ(umap.begin(), umap.end());
}

// const 반복자 연산 테스트 - 상수 반복자 기능 확인
TEST_F(UnorderedMapTest, ConstIterator) {
    UnorderedMap<int, std::string> umap;
    populateMap(umap);
    
    // const 참조 생성
    const UnorderedMap<int, std::string>& const_umap = umap;
    
    // begin/end 테스트
    auto it = const_umap.begin();
    EXPECT_NE(it, const_umap.end());
    
    // const_iterator 접근 테스트
    EXPECT_EQ(it->first, (*it).first);
    
    // const_iterator를 사용하여 요소 수 세기
    size_t count = 0;
    for (auto it = const_umap.begin(); it != const_umap.end(); ++it) {
        count++;
    }
    EXPECT_EQ(5, count);
    
    // cbegin/cend 테스트
    for (auto it = const_umap.cbegin(); it != const_umap.cend(); ++it) {
        EXPECT_GE(5, it->first);  // 모든 키는 <= 5 여야 함
    }
}

// 버킷 인터페이스 테스트 - 해시 테이블 내부 구조 접근 기능 확인
TEST_F(UnorderedMapTest, BucketInterface) {
    UnorderedMap<int, std::string> umap;
    populateMap(umap);
    
    // 버킷 수 테스트
    EXPECT_EQ(8, umap.bucket_count());
    
    // 최대 버킷 수 테스트
    EXPECT_GT(umap.max_bucket_count(), 0);
    
    // bucket(key) 테스트
    for (int i = 1; i <= 5; i++) {
        size_t bucket_idx = umap.bucket(i);
        EXPECT_LT(bucket_idx, umap.bucket_count());
    }
    
    // bucket_size 테스트
    size_t total_bucket_sizes = 0;
    for (size_t i = 0; i < umap.bucket_count(); i++) {
        total_bucket_sizes += umap.bucket_size(i);
    }
    EXPECT_EQ(umap.size(), total_bucket_sizes);
}

// 해시 함수 및 키 비교 테스트 - 해시 함수와 키 비교 함수 동작 확인
TEST_F(UnorderedMapTest, HashAndEqualityFunctions) {
    UnorderedMap<int, std::string> umap;
    
    // hash_function 테스트
    auto hash_func = umap.hash_function();
    EXPECT_EQ(hash_func(5), hash_func(5));  // 같은 값에 대한 해시는 같아야 함
    
    // key_eq 테스트
    auto key_eq = umap.key_eq();
    EXPECT_TRUE(key_eq(5, 5));
    EXPECT_FALSE(key_eq(5, 6));
    
    // 사용자 정의 해시 및 비교 함수 테스트
    struct CustomHash {
        size_t operator()(const std::string& key) const {
            // 간단한 사용자 정의 해시: ASCII 값의 합
            size_t hash = 0;
            for (char c : key) {
                hash += static_cast<size_t>(c);
            }
            return hash;
        }
    };
    
    struct CustomEqual {
        bool operator()(const std::string& lhs, const std::string& rhs) const {
            // 대소문자 구분 없는 동등성
            if (lhs.size() != rhs.size()) return false;
            for (size_t i = 0; i < lhs.size(); ++i) {
                if (std::tolower(lhs[i]) != std::tolower(rhs[i])) return false;
            }
            return true;
        }
    };
    
    UnorderedMap<std::string, int, CustomHash, CustomEqual> custom_umap;
    custom_umap["test"] = 1;
    
    // 사용자 정의 해시 확인
    auto custom_hash = custom_umap.hash_function();
    EXPECT_EQ(custom_hash("abc"), custom_hash("abc"));
    
    // 사용자 정의 동등성 확인
    auto custom_eq = custom_umap.key_eq();
    EXPECT_TRUE(custom_eq("Test", "test"));  // 대소문자 구분 없이 동일해야 함
    EXPECT_FALSE(custom_eq("Test", "tests"));  // 길이가 다름
    
    // 맵 동작 확인
    custom_umap["TEST"] = 2;  // 대소문자 구분 없는 동등성으로 인해 "test"를 업데이트해야 함
    EXPECT_EQ(1, custom_umap.size());
    EXPECT_EQ(2, custom_umap["test"]);
}

// 스왑 테스트 - 두 맵 간의 내용 교환 기능 확인
TEST_F(UnorderedMapTest, SwapOperation) {
    UnorderedMap<int, std::string> umap1;
    umap1[1] = "one";
    umap1[2] = "two";
    
    UnorderedMap<int, std::string> umap2;
    umap2[3] = "three";
    umap2[4] = "four";
    umap2[5] = "five";
    
    EXPECT_EQ(2, umap1.size());
    EXPECT_EQ(3, umap2.size());
    
    // 스왑 수행
    umap1.swap(umap2);
    
    // 스왑 후 크기 확인
    EXPECT_EQ(3, umap1.size());
    EXPECT_EQ(2, umap2.size());
    
    // 스왑 후 내용 확인
    EXPECT_EQ("three", umap1[3]);
    EXPECT_EQ("four", umap1[4]);
    EXPECT_EQ("five", umap1[5]);
    
    EXPECT_EQ("one", umap2[1]);
    EXPECT_EQ("two", umap2[2]);
}

// 엣지 케이스 테스트 - 경계 조건 동작 확인
TEST_F(UnorderedMapTest, EdgeCases) {
    UnorderedMap<int, std::string> umap;
    
    // 빈 맵 동작 테스트
    EXPECT_TRUE(umap.empty());
    EXPECT_EQ(umap.begin(), umap.end());
    EXPECT_EQ(0, umap.count(1));
    EXPECT_EQ(umap.end(), umap.find(1));
    EXPECT_EQ(umap.end(), umap.equal_range(1).first);
    EXPECT_EQ(umap.end(), umap.equal_range(1).second);
    
    // 단일 요소 맵
    umap[1] = "one";
    EXPECT_FALSE(umap.empty());
    EXPECT_NE(umap.begin(), umap.end());
    auto it = umap.begin();
    ++it;
    EXPECT_EQ(it, umap.end());
    
    // 유일한 요소 삭제
    umap.erase(1);
    EXPECT_TRUE(umap.empty());
    EXPECT_EQ(umap.begin(), umap.end());
}

// 리소스 소유 타입 테스트 - unique_ptr과 같은 이동 전용 타입 지원 확인
TEST_F(UnorderedMapTest, ResourceOwning) {
    // unique_ptr 테스트
    UnorderedMap<int, std::unique_ptr<int>> resource_map;
    
    // 소유 리소스 삽입
    for (int i = 0; i < 10; i++) {
        resource_map[i] = std::make_unique<int>(i * 10);
    }
    
    EXPECT_EQ(10, resource_map.size());
    
    // 접근 및 확인
    for (int i = 0; i < 10; i++) {
        ASSERT_NE(nullptr, resource_map[i]);
        EXPECT_EQ(i * 10, *resource_map[i]);
    }
    
    // unique_ptr의 이동 테스트
    auto ptr = std::make_unique<int>(100);
    int* raw_ptr = ptr.get();
    resource_map[10] = std::move(ptr);
    EXPECT_EQ(nullptr, ptr);  // 원본 포인터는 이동 후 null이어야 함
    EXPECT_EQ(raw_ptr, resource_map[10].get());  // 같은 포인터를 가리켜야 함
    
    // 절반 삭제
    for (int i = 0; i < 5; i++) {
        resource_map.erase(i);
    }
    
    EXPECT_EQ(6, resource_map.size());
    
    // clear는 모든 리소스를 적절히 해제해야 함
    resource_map.clear();
    EXPECT_EQ(0, resource_map.size());
}

// unique_ptr<string> 테스트 - 복잡한 이동 전용 타입 지원 확인
TEST_F(UnorderedMapTest, UniquePtrString) {
    // std::unique_ptr<std::string>을 사용하여 move-only 타입 테스트
    using ValueType = std::unique_ptr<std::string>;
    UnorderedMap<int, ValueType> move_map;
    
    // 요소 추가
    move_map[1] = std::make_unique<std::string>("one");
    move_map[2] = std::make_unique<std::string>("two");
    
    // emplace로 요소 추가
    auto ptr = std::make_unique<std::string>("three");
    move_map.emplace(3, std::move(ptr));
    
    // 이동 생성 테스트
    ValueType val = std::make_unique<std::string>("four");
    move_map[4] = std::move(val);
    EXPECT_EQ(nullptr, val);  // 원본은 이동 후 null이어야 함
    
    // 값 확인
    EXPECT_EQ("one", *move_map[1]);
    EXPECT_EQ("two", *move_map[2]);
    EXPECT_EQ("three", *move_map[3]);
    EXPECT_EQ("four", *move_map[4]);
    
    // 모든 요소에 접근하고 수정
    for (auto& pair : move_map) {
        *pair.second += "_modified";
    }
    
    // 수정된 값 확인
    EXPECT_EQ("one_modified", *move_map[1]);
    EXPECT_EQ("two_modified", *move_map[2]);
    
    // 맵 지우기
    move_map.clear();
    EXPECT_EQ(0, move_map.size());
}

//-----------------------------------------------------------------------------
// 메인 함수 - 모든 테스트 실행 및 추가 테스트
//-----------------------------------------------------------------------------
int main(int argc, char **argv) {
    // Google Test 초기화
    ::testing::InitGoogleTest(&argc, argv);
    // 모든 테스트 실행
    RUN_ALL_TESTS();

    // 추가 테스트 - 기본 작업
    std::cout << "=== Basic Operations ===" << std::endl;
    UnorderedMap<int, std::string> umap;
    umap.insert(std::pair<const int, std::string>(1, "one"));
    umap.insert(std::pair<const int, std::string>(2, "two"));
    umap[3] = "three";

    std::cout << "Size: " << umap.size() << std::endl;
    std::cout << "Key 2: " << umap.at(2) << std::endl;

    // Test count function
    std::cout << "Count of key 2: " << umap.count(2) << std::endl;
    std::cout << "Count of key 5: " << umap.count(5) << std::endl;

    // 반복자 순회
    std::cout << "\n=== Iterator Traversal ===" << std::endl;
    for (UnorderedMap<int, std::string>::iterator it = umap.begin(); it != umap.end(); ++it) {
        std::cout << it->first << " : " << it->second << std::endl;
    }

    // const 반복자 테스트
    std::cout << "\n=== Const Iterator Traversal ===" << std::endl;
    const UnorderedMap<int, std::string>& const_umap = umap;
    for (UnorderedMap<int, std::string>::const_iterator it = const_umap.begin(); it != const_umap.end(); ++it) {
        std::cout << it->first << " : " << it->second << std::endl;
    }

    // Emplace 및 equal_range
    std::cout << "\n=== Emplace and Equal Range ===" << std::endl;
    // Emplace 테스트
    auto result = umap.emplace(4, "four");
    std::cout << "Emplace result - success: " << (result.second ? "true" : "false") 
              << ", key: " << result.first->first << ", value: " << result.first->second << std::endl;
    
    // 중복 emplace 테스트
    result = umap.emplace(4, "another_four");
    std::cout << "Duplicate emplace - success: " << (result.second ? "true" : "false") 
              << ", key: " << result.first->first << ", value: " << result.first->second << std::endl;

    // equal_range 테스트
    auto range = umap.equal_range(3);
    std::cout << "Equal range for key 3:" << std::endl;
    for (auto it = range.first; it != range.second; ++it) {
        std::cout << "  " << it->first << " : " << it->second << std::endl;
    }

    // 삭제 작업
    std::cout << "\n=== Erase Operations ===" << std::endl;
    std::cout << "Before erasing key 1, size: " << umap.size() << std::endl;
    size_t erased_count = umap.erase(1);
    std::cout << "After erasing key 1, size: " << umap.size() << ", erased count: " << erased_count << std::endl;
    
    // 반복자로 삭제
    auto it_to_erase = umap.find(3);
    if (it_to_erase != umap.end()) {
        std::cout << "Erasing key 3 using iterator" << std::endl;
        auto next_it = umap.erase(it_to_erase);
        std::cout << "After erasing, size: " << umap.size() << std::endl;
        if (next_it != umap.end()) {
            std::cout << "Next element after erase: " << next_it->first << " : " << next_it->second << std::endl;
        }
    }

    // 버킷 인터페이스
    std::cout << "\n=== Bucket Interface ===" << std::endl;
    std::cout << "Bucket count: " << umap.bucket_count() << std::endl;
    std::cout << "Max bucket count: " << umap.max_bucket_count() << std::endl;
    
    // 버킷 크기 확인
    std::cout << "Bucket sizes:" << std::endl;
    for (size_t i = 0; i < umap.bucket_count(); ++i) {
        std::cout << "  Bucket " << i << ": " << umap.bucket_size(i) << std::endl;
    }
    
    // 특정 키가 어떤 버킷에 있는지 확인
    int test_key = 4;
    std::cout << "Key " << test_key << " is in bucket: " << umap.bucket(test_key) << std::endl;

    // 스왑 기능
    std::cout << "\n=== Swap Functionality ===" << std::endl;
    UnorderedMap<int, std::string> umap2;
    umap2[5] = "five";
    umap2[6] = "six";
    
    std::cout << "Before swap:" << std::endl;
    std::cout << "  umap size: " << umap.size() << std::endl;
    std::cout << "  umap2 size: " << umap2.size() << std::endl;
    
    umap.swap(umap2);
    
    std::cout << "After swap:" << std::endl;
    std::cout << "  umap size: " << umap.size() << std::endl;
    std::cout << "  umap2 size: " << umap2.size() << std::endl;
    
    std::cout << "umap contents after swap:" << std::endl;
    for (const auto& pair : umap) {
        std::cout << "  " << pair.first << " : " << pair.second << std::endl;
    }

    // 사용자 정의 해시 및 비교 함수
    std::cout << "\n=== Custom Hash and Equality Function ===" << std::endl;
    struct CustomHash {
        size_t operator()(const std::string& key) const {
            // 간단한 사용자 정의 해시: ASCII 값의 합
            size_t hash = 0;
            for (char c : key) {
                hash += static_cast<size_t>(c);
            }
            return hash;
        }
    };
    
    struct CustomEqual {
        bool operator()(const std::string& lhs, const std::string& rhs) const {
            // 대소문자 구분 없는 동등성
            if (lhs.size() != rhs.size()) return false;
            for (size_t i = 0; i < lhs.size(); ++i) {
                if (std::tolower(lhs[i]) != std::tolower(rhs[i])) return false;
            }
            return true;
        }
    };
    
    UnorderedMap<std::string, int, CustomHash, CustomEqual> custom_umap;
    custom_umap["test"] = 1;
    custom_umap["TEST"] = 2;  // 대소문자 구분 없는 동등성으로 인해 "test" 업데이트
    custom_umap["another"] = 3;
    
    std::cout << "Custom unordered map size: " << custom_umap.size() << std::endl;
    std::cout << "custom_umap[\"test\"]: " << custom_umap["test"] << std::endl;
    std::cout << "Custom unordered map contents:" << std::endl;
    for (const auto& pair : custom_umap) {
        std::cout << "  " << pair.first << " : " << pair.second << std::endl;
    }

    // 엣지 케이스
    std::cout << "\n=== Edge Cases ===" << std::endl;
    // 빈 맵
    UnorderedMap<int, std::string> empty_map;
    std::cout << "Empty map size: " << empty_map.size() << std::endl;
    std::cout << "Empty map begin() == end(): " << (empty_map.begin() == empty_map.end() ? "true" : "false") << std::endl;
    
    // 존재하지 않는 키에 at() 사용 - 예외 발생해야 함
    try {
        empty_map.at(10);
        std::cout << "This should not be printed!" << std::endl;
    } catch (const std::out_of_range& ex) {
        std::cout << "Caught expected exception: " << ex.what() << std::endl;
    }
    
    // 존재하지 않는 키에 operator[] 사용 - 기본값 삽입해야 함
    std::cout << "empty_map[10]: " << empty_map[10] << std::endl;
    std::cout << "After using operator[], size: " << empty_map.size() << std::endl;

    // hash_function 및 key_eq 접근자 테스트
    std::cout << "\n=== Hash Function and Key Equality ===" << std::endl;
    auto hash_func = custom_umap.hash_function();
    auto key_eq_func = custom_umap.key_eq();
    
    std::string test_str1 = "example";
    std::string test_str2 = "EXAMPLE";
    
    std::cout << "Hash of \"" << test_str1 << "\": " << hash_func(test_str1) << std::endl;
    std::cout << "Hash of \"" << test_str2 << "\": " << hash_func(test_str2) << std::endl;
    std::cout << "Are \"" << test_str1 << "\" and \"" << test_str2 
              << "\" equal: " << (key_eq_func(test_str1, test_str2) ? "true" : "false") << std::endl;

    // 마지막 정리: 모든 맵 clear
    std::cout << "\n=== Final Cleanup ===" << std::endl;
    umap.clear();
    umap2.clear();
    custom_umap.clear();
    empty_map.clear();
    std::cout << "After clearing, all maps empty:" << std::endl;
    std::cout << "  umap: " << (umap.empty() ? "true" : "false") << std::endl;
    std::cout << "  umap2: " << (umap2.empty() ? "true" : "false") << std::endl;
    std::cout << "  custom_umap: " << (custom_umap.empty() ? "true" : "false") << std::endl;
    std::cout << "  empty_map: " << (empty_map.empty() ? "true" : "false") << std::endl;
    
    return 0;
}