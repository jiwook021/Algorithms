#include <iostream>
#include <vector>
#include <list>
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
// UnorderedMap 클래스 템플릿 (모든 타입을 직접 명시)
// Key         : 키 타입
// T           : 값 타입
// Hash        : 해시 함수 (기본: std::hash<Key>)
// KeyEqual    : 키 비교 함수 (기본: std::equal_to<Key>)
//
// ※ 내부에서는 버킷 관리를 위해 std::vector와 std::list를 사용함.
// -----------------------------------------------------------------------------
template <typename Key, typename T, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key> >
class UnorderedMap {
private:
    // 기본 버킷 수
    static const size_t default_bucket_count = 8;

    // 버킷은 std::list<std::pair<const Key, T> >로 구현
    std::vector< std::list< std::pair<const Key, T> > > buckets;

    // 저장된 전체 요소 수
    size_t num_elements;

    // 해시 함수와 키 비교 객체 (멤버 함수 key_eq()와 충돌하지 않도록 이름 변경)
    Hash hash_func;
    KeyEqual key_equal_obj;

    // 내부: 주어진 키의 버킷 인덱스 계산
    size_t bucket_index(const Key & key) const {
        return hash_func(key) % buckets.size();
    }

public:
    // ======================================================
    // 생성자 / 소멸자
    // ======================================================
    UnorderedMap()
        : buckets(default_bucket_count),
          num_elements(0),
          hash_func(Hash()),
          key_equal_obj(KeyEqual())
    {
    }

    ~UnorderedMap() {
        clear();
    }

    // ======================================================
    // Capacity 관련 멤버 함수
    // ======================================================
    bool empty() const {
        return num_elements == 0;
    }

    size_t size() const {
        return num_elements;
    }

    size_t max_size() const {
        return std::numeric_limits<size_t>::max();
    }

    // ======================================================
    // Iterator 구현
    // ======================================================
    // iterator 클래스 (non-const)
    class iterator {
    private:
        UnorderedMap * map_ptr; // UnorderedMap 객체 포인터
        size_t bucket_idx;      // 현재 버킷 인덱스
        // 현재 버킷 내 반복자 (dependent type 앞에 typename 추가)
        typename std::list< std::pair<const Key, T> >::iterator bucket_iter;

        // 빈 버킷을 건너뛰는 함수
        void advance_to_valid() {
            while (map_ptr && bucket_idx < map_ptr->buckets.size() &&
                   bucket_iter == map_ptr->buckets[bucket_idx].end())
            {
                ++bucket_idx;
                if (bucket_idx < map_ptr->buckets.size())
                    bucket_iter = map_ptr->buckets[bucket_idx].begin();
            }
        }

        // UnorderedMap이 private 멤버에 접근할 수 있도록 friend 선언
        friend class UnorderedMap;
        // const_iterator와 비교를 위해 friend 선언
        friend class const_iterator;
    public:
        iterator() : map_ptr(nullptr), bucket_idx(0) {
        }

        iterator(UnorderedMap * m, size_t idx, typename std::list< std::pair<const Key, T> >::iterator it)
            : map_ptr(m), bucket_idx(idx), bucket_iter(it)
        {
            if (map_ptr)
                advance_to_valid();
        }

        std::pair<const Key, T> & operator*() const {
            return *bucket_iter;
        }

        std::pair<const Key, T> * operator->() const {
            return &(*bucket_iter);
        }

        iterator & operator++() {
            ++bucket_iter;
            advance_to_valid();
            return *this;
        }

        iterator operator++(int) {
            iterator temp = *this;
            ++(*this);
            return temp;
        }

        // 수정된 operator==: 만약 두 반복자 모두 bucket_idx가 buckets.size()라면 end 상태로 간주함.
        bool operator==(const iterator & other) const {
            if (map_ptr != other.map_ptr)
                return false;
            if (bucket_idx == map_ptr->buckets.size() && other.bucket_idx == map_ptr->buckets.size())
                return true;
            return bucket_idx == other.bucket_idx && bucket_iter == other.bucket_iter;
        }

        bool operator!=(const iterator & other) const {
            return !(*this == other);
        }
    };

    // const_iterator 클래스 (읽기 전용)
    class const_iterator {
    private:
        const UnorderedMap * map_ptr; // UnorderedMap 객체 포인터 (const)
        size_t bucket_idx;            // 현재 버킷 인덱스
        // 현재 버킷 내 const 반복자 (dependent type 앞에 typename 추가)
        typename std::list< std::pair<const Key, T> >::const_iterator bucket_iter;

        void advance_to_valid() {
            while (map_ptr && bucket_idx < map_ptr->buckets.size() &&
                   bucket_iter == map_ptr->buckets[bucket_idx].end())
            {
                ++bucket_idx;
                if (bucket_idx < map_ptr->buckets.size())
                    bucket_iter = map_ptr->buckets[bucket_idx].begin();
            }
        }

        friend class UnorderedMap;
    public:
        const_iterator() : map_ptr(nullptr), bucket_idx(0) {
        }

        const_iterator(const UnorderedMap * m, size_t idx, typename std::list< std::pair<const Key, T> >::const_iterator it)
            : map_ptr(m), bucket_idx(idx), bucket_iter(it)
        {
            if (map_ptr)
                advance_to_valid();
        }

        const std::pair<const Key, T> & operator*() const {
            return *bucket_iter;
        }

        const std::pair<const Key, T> * operator->() const {
            return &(*bucket_iter);
        }

        const_iterator & operator++() {
            ++bucket_iter;
            advance_to_valid();
            return *this;
        }

        const_iterator operator++(int) {
            const_iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator==(const const_iterator & other) const {
            if (map_ptr != other.map_ptr)
                return false;
            if (bucket_idx == map_ptr->buckets.size() && other.bucket_idx == map_ptr->buckets.size())
                return true;
            return bucket_idx == other.bucket_idx && bucket_iter == other.bucket_iter;
        }

        bool operator!=(const const_iterator & other) const {
            return !(*this == other);
        }
    };

    // ======================================================
    // Iterator 접근자
    // ======================================================
    iterator begin() {
        for (size_t i = 0; i < buckets.size(); ++i) {
            if (!buckets[i].empty())
                return iterator(this, i, buckets[i].begin());
        }
        return end();
    }

    iterator end() {
        // end()는 bucket_idx가 buckets.size()인 상태로 생성
        return iterator(this, buckets.size(), typename std::list< std::pair<const Key, T> >::iterator());
    }

    const_iterator begin() const {
        for (size_t i = 0; i < buckets.size(); ++i) {
            if (!buckets[i].empty())
                return const_iterator(this, i, buckets[i].begin());
        }
        return end();
    }

    const_iterator end() const {
        return const_iterator(this, buckets.size(), typename std::list< std::pair<const Key, T> >::const_iterator());
    }

    const_iterator cbegin() const {
        return begin();
    }

    const_iterator cend() const {
        return end();
    }

    // ======================================================
    // Element Access
    // ======================================================
    // operator[] : 키가 없으면 기본값으로 새 요소 생성
    // 수정: piecewise_construct 사용하여 in-place 생성
    T & operator[](const Key & key) {
        iterator it = find(key);
        if (it == end()) {
            // 키가 없으면 새 요소 생성
            size_t idx = bucket_index(key);
            
            // std::piecewise_construct를 사용하여 요소 in-place 생성
            buckets[idx].emplace_back(
                std::piecewise_construct,
                std::forward_as_tuple(key),  // key는 const Key&로 전달
                std::forward_as_tuple()      // T()의 인자 없음
            );
            
            ++num_elements;
            auto list_it = buckets[idx].end();
            --list_it;
            return iterator(this, idx, list_it)->second;
        }
        return it->second;
    }

    // 수정: rvalue 참조 지원 추가
    T & operator[](Key && key) {
        iterator it = find(key);
        if (it == end()) {
            // 키가 없으면 새 요소 생성
            size_t idx = bucket_index(key);
            
            // std::piecewise_construct를 사용하여 요소 in-place 생성
            buckets[idx].emplace_back(
                std::piecewise_construct,
                std::forward_as_tuple(std::move(key)),  // key는 Key&&로 이동
                std::forward_as_tuple()                 // T()의 인자 없음
            );
            
            ++num_elements;
            auto list_it = buckets[idx].end();
            --list_it;
            return iterator(this, idx, list_it)->second;
        }
        return it->second;
    }

    // at() : 키가 없으면 std::out_of_range 예외 발생
    T & at(const Key & key) {
        iterator it = find(key);
        if (it == end())
            throw std::out_of_range("Key not found");
        return it->second;
    }

    const T & at(const Key & key) const {
        const_iterator it = find(key);
        if (it == end())
            throw std::out_of_range("Key not found");
        return it->second;
    }

    // ======================================================
    // Lookup
    // ======================================================
    iterator find(const Key & key) {
        size_t idx = bucket_index(key);
        typename std::list< std::pair<const Key, T> >::iterator it = buckets[idx].begin();
        for (; it != buckets[idx].end(); ++it) {
            if (key_equal_obj(it->first, key))
                return iterator(this, idx, it);
        }
        return end();
    }

    const_iterator find(const Key & key) const {
        size_t idx = bucket_index(key);
        typename std::list< std::pair<const Key, T> >::const_iterator it = buckets[idx].begin();
        for (; it != buckets[idx].end(); ++it) {
            if (key_equal_obj(it->first, key))
                return const_iterator(this, idx, it);
        }
        return end();
    }

    size_t count(const Key & key) const {
        return (find(key) == end()) ? 0 : 1;
    }

    std::pair<iterator, iterator> equal_range(const Key & key) {
        iterator it = find(key);
        if (it == end())
            return std::pair<iterator, iterator>(end(), end());
        iterator next = it;
        ++next;
        return std::pair<iterator, iterator>(it, next);
    }

    std::pair<const_iterator, const_iterator> equal_range(const Key & key) const {
        const_iterator it = find(key);
        if (it == end())
            return std::pair<const_iterator, const_iterator>(end(), end());
        const_iterator next = it;
        ++next;
        return std::pair<const_iterator, const_iterator>(it, next);
    }

    // ======================================================
    // Modifiers
    // ======================================================
    // emplace : 전달받은 인자를 이용해 새 요소 생성 후 삽입
    template <typename... Args>
    std::pair<iterator, bool> emplace(Args&&... args) {
        // 먼저 임시 pair 객체 생성
        auto temp_pair = std::pair<const Key, T>(std::forward<Args>(args)...);
        const Key& key = temp_pair.first;
        
        // 이미 존재하는지 확인
        size_t idx = bucket_index(key);
        typename std::list< std::pair<const Key, T> >::iterator it = buckets[idx].begin();
        for (; it != buckets[idx].end(); ++it) {
            if (key_equal_obj(it->first, key))
                return std::pair<iterator, bool>(iterator(this, idx, it), false);
        }
        
        // piecewise_construct를 사용하여 요소 in-place 생성
        buckets[idx].emplace_back(std::move(temp_pair));
        ++num_elements;
        it = buckets[idx].end();
        --it;
        return std::pair<iterator, bool>(iterator(this, idx, it), true);
    }

    // insert : lvalue 기반 삽입
    std::pair<iterator, bool> insert(const std::pair<const Key, T> & val) {
        size_t idx = bucket_index(val.first);
        typename std::list< std::pair<const Key, T> >::iterator it = buckets[idx].begin();
        for (; it != buckets[idx].end(); ++it) {
            if (key_equal_obj(it->first, val.first))
                return std::pair<iterator, bool>(iterator(this, idx, it), false);
        }
        buckets[idx].push_back(val);
        ++num_elements;
        it = buckets[idx].end();
        --it;
        return std::pair<iterator, bool>(iterator(this, idx, it), true);
    }

    // insert : rvalue 기반 삽입 (move semantics)
    std::pair<iterator, bool> insert(std::pair<const Key, T> && val) {
        size_t idx = bucket_index(val.first);
        typename std::list< std::pair<const Key, T> >::iterator it = buckets[idx].begin();
        for (; it != buckets[idx].end(); ++it) {
            if (key_equal_obj(it->first, val.first))
                return std::pair<iterator, bool>(iterator(this, idx, it), false);
        }
        buckets[idx].push_back(std::move(val));
        ++num_elements;
        it = buckets[idx].end();
        --it;
        return std::pair<iterator, bool>(iterator(this, idx, it), true);
    }

    // erase : 키 기반 삭제, 존재하면 1, 아니면 0 반환
    size_t erase(const Key & key) {
        size_t idx = bucket_index(key);
        typename std::list< std::pair<const Key, T> >::iterator it = buckets[idx].begin();
        for (; it != buckets[idx].end(); ) {
            if (key_equal_obj(it->first, key)) {
                it = buckets[idx].erase(it);
                --num_elements;
                return 1;
            } else {
                ++it;
            }
        }
        return 0;
    }

    // erase : iterator 기반 삭제
    iterator erase(iterator pos) {
        if (pos == end())
            return pos;
        size_t idx = pos.bucket_idx;
        typename std::list< std::pair<const Key, T> >::iterator it = buckets[idx].erase(pos.bucket_iter);
        --num_elements;
        return iterator(this, idx, it);
    }

    // clear : 모든 요소 삭제
    void clear() {
        for (size_t i = 0; i < buckets.size(); ++i) {
            buckets[i].clear();
        }
        num_elements = 0;
    }

    // swap : 다른 UnorderedMap과 내용 교환
    void swap(UnorderedMap & other) {
        std::swap(buckets, other.buckets);
        std::swap(num_elements, other.num_elements);
        std::swap(hash_func, other.hash_func);
        std::swap(key_equal_obj, other.key_equal_obj);
    }

    // ======================================================
    // Bucket 인터페이스
    // ======================================================
    size_t bucket_count() const {
        return buckets.size();
    }

    size_t max_bucket_count() const {
        return std::numeric_limits<size_t>::max();
    }

    size_t bucket_size(size_t n) const {
        return (n < buckets.size() ? buckets[n].size() : 0);
    }

    size_t bucket(const Key & key) const {
        return bucket_index(key);
    }

    // ======================================================
    // Hash 및 키 비교 객체 접근자
    // ======================================================
    Hash hash_function() const {
        return hash_func;
    }

    KeyEqual key_eq() const {
        return key_equal_obj;
    }
};

//-----------------------------------------------------------------------------
// Google Test 테스트 케이스
//-----------------------------------------------------------------------------

// 기본 테스트 클래스 설정
class UnorderedMapTest : public ::testing::Test {
protected:
    // 테스트 전에 실행되는 메서드
    void SetUp() override {
        // 테스트 전 설정 코드
    }

    // 테스트 후에 실행되는 메서드
    void TearDown() override {
        // 테스트 후 정리 코드
    }

    // 맵에 테스트 데이터를 채우는 헬퍼 함수
    void populateMap(UnorderedMap<int, std::string>& map) {
        map.insert(std::pair<const int, std::string>(1, "one"));
        map.insert(std::pair<const int, std::string>(2, "two"));
        map[3] = "three";
        map[4] = "four";
        map[5] = "five";
    }
};

// 생성자 및 기본 용량 메서드 테스트
TEST_F(UnorderedMapTest, ConstructorAndCapacity) {
    UnorderedMap<int, std::string> umap;
    EXPECT_TRUE(umap.empty());
    EXPECT_EQ(0, umap.size());
    
    // max_size는 큰 값이어야 함
    EXPECT_GT(umap.max_size(), 0);
    
    // 초기 버킷 수는 기본값과 일치해야 함
    EXPECT_EQ(8, umap.bucket_count());
}

// 삽입 및 요소 접근 테스트
TEST_F(UnorderedMapTest, InsertionAndAccess) {
    UnorderedMap<int, std::string> umap;
    
    // insert 테스트
    auto result = umap.insert(std::pair<const int, std::string>(1, "one"));
    EXPECT_TRUE(result.second); // 성공적으로 삽입되어야 함
    EXPECT_EQ("one", result.first->second);
    EXPECT_EQ(1, umap.size());
    
    // 중복 키 insert 테스트
    result = umap.insert(std::pair<const int, std::string>(1, "another_one"));
    EXPECT_FALSE(result.second); // 삽입되지 않아야 함
    EXPECT_EQ("one", result.first->second); // 값은 변경되지 않아야 함
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
    EXPECT_EQ("", umap[3]); // 기본 초기화된 문자열
    EXPECT_EQ(3, umap.size());
    
    EXPECT_EQ("", umap[4]); 
    EXPECT_EQ(4, umap.size());
    

    // rvalue key 테스트
    int temp_key = 4;
    umap[std::move(temp_key)] = "four";
    EXPECT_EQ("four", umap[4]);
    EXPECT_EQ(4, umap.size());
}

// emplace 테스트
TEST_F(UnorderedMapTest, Emplace) {
    UnorderedMap<int, std::string> umap;
    
    // 기본 emplace 테스트
    auto result = umap.emplace(1, "one");
    EXPECT_TRUE(result.second);
    EXPECT_EQ("one", result.first->second);
    
    // 중복 키 emplace 테스트
    result = umap.emplace(1, "another");
    EXPECT_FALSE(result.second);
    EXPECT_EQ("one", result.first->second); // 값은 변경되지 않아야 함
    
    // 새로운 키 emplace 테스트
    result = umap.emplace(2, "two");
    EXPECT_TRUE(result.second);
    EXPECT_EQ("two", result.first->second);
    
    EXPECT_EQ(2, umap.size());
}

// find 및 count 테스트
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

// 삭제 연산 테스트
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
    EXPECT_NE(umap.end(), next_it); // 유효한 요소를 가리켜야 함
    
    // clear 테스트
    umap.clear();
    EXPECT_TRUE(umap.empty());
    EXPECT_EQ(0, umap.size());
    EXPECT_EQ(umap.begin(), umap.end());
}

// const 반복자 연산 테스트
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
        EXPECT_GE(5, it->first); // 모든 키는 <= 5 여야 함
    }
}

// 버킷 인터페이스 테스트
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

// 해시 함수 및 키 비교 테스트
TEST_F(UnorderedMapTest, HashAndEqualityFunctions) {
    UnorderedMap<int, std::string> umap;
    
    // hash_function 테스트
    auto hash_func = umap.hash_function();
    EXPECT_EQ(hash_func(5), hash_func(5)); // 같은 값에 대한 해시는 같아야 함
    
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
    EXPECT_TRUE(custom_eq("Test", "test")); // 대소문자 구분 없이 동일해야 함
    EXPECT_FALSE(custom_eq("Test", "tests")); // 길이가 다름
    
    // 맵 동작 확인
    custom_umap["TEST"] = 2; // 대소문자 구분 없는 동등성으로 인해 "test"를 업데이트해야 함
    EXPECT_EQ(1, custom_umap.size());
    EXPECT_EQ(2, custom_umap["test"]);
}

// 스왑 테스트
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

// 엣지 케이스 테스트
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

// 리소스 소유 타입 테스트 (unique_ptr)
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
    EXPECT_EQ(nullptr, ptr); // 원본 포인터는 이동 후 null이어야 함
    EXPECT_EQ(raw_ptr, resource_map[10].get()); // 같은 포인터를 가리켜야 함
    
    // 절반 삭제
    for (int i = 0; i < 5; i++) {
        resource_map.erase(i);
    }
    
    EXPECT_EQ(6, resource_map.size());
    
    // clear는 모든 리소스를 적절히 해제해야 함
    resource_map.clear();
    EXPECT_EQ(0, resource_map.size());
}

// unique_ptr<string> 테스트
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
    EXPECT_EQ(nullptr, val); // 원본은 이동 후 null이어야 함
    
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
// 메인 함수 - 모든 테스트 실행
//-----------------------------------------------------------------------------
int main(int argc, char **argv) {
    // Google Test 초기화
    ::testing::InitGoogleTest(&argc, argv);
    // 모든 테스트 실행
    RUN_ALL_TESTS();


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

    // Section 2: Iteration
    std::cout << "\n=== Iterator Traversal ===" << std::endl;
    for (UnorderedMap<int, std::string>::iterator it = umap.begin(); it != umap.end(); ++it) {
        std::cout << it->first << " : " << it->second << std::endl;
    }

    // Test const iteration
    std::cout << "\n=== Const Iterator Traversal ===" << std::endl;
    const UnorderedMap<int, std::string>& const_umap = umap;
    for (UnorderedMap<int, std::string>::const_iterator it = const_umap.begin(); it != const_umap.end(); ++it) {
        std::cout << it->first << " : " << it->second << std::endl;
    }

    // Section 3: Emplace and equal_range
    std::cout << "\n=== Emplace and Equal Range ===" << std::endl;
    // Test emplace
    auto result = umap.emplace(4, "four");
    std::cout << "Emplace result - success: " << (result.second ? "true" : "false") 
              << ", key: " << result.first->first << ", value: " << result.first->second << std::endl;
    
    // Test duplicate emplace
    result = umap.emplace(4, "another_four");
    std::cout << "Duplicate emplace - success: " << (result.second ? "true" : "false") 
              << ", key: " << result.first->first << ", value: " << result.first->second << std::endl;

    // Test equal_range
    auto range = umap.equal_range(3);
    std::cout << "Equal range for key 3:" << std::endl;
    for (auto it = range.first; it != range.second; ++it) {
        std::cout << "  " << it->first << " : " << it->second << std::endl;
    }

    // Section 4: Erase operations
    std::cout << "\n=== Erase Operations ===" << std::endl;
    std::cout << "Before erasing key 1, size: " << umap.size() << std::endl;
    size_t erased_count = umap.erase(1);
    std::cout << "After erasing key 1, size: " << umap.size() << ", erased count: " << erased_count << std::endl;
    
    // Erase using iterator
    auto it_to_erase = umap.find(3);
    if (it_to_erase != umap.end()) {
        std::cout << "Erasing key 3 using iterator" << std::endl;
        auto next_it = umap.erase(it_to_erase);
        std::cout << "After erasing, size: " << umap.size() << std::endl;
        if (next_it != umap.end()) {
            std::cout << "Next element after erase: " << next_it->first << " : " << next_it->second << std::endl;
        }
    }

    // Section 5: Bucket interface
    std::cout << "\n=== Bucket Interface ===" << std::endl;
    std::cout << "Bucket count: " << umap.bucket_count() << std::endl;
    std::cout << "Max bucket count: " << umap.max_bucket_count() << std::endl;
    
    // Check bucket sizes
    std::cout << "Bucket sizes:" << std::endl;
    for (size_t i = 0; i < umap.bucket_count(); ++i) {
        std::cout << "  Bucket " << i << ": " << umap.bucket_size(i) << std::endl;
    }
    
    // Find which bucket a key is in
    int test_key = 4;
    std::cout << "Key " << test_key << " is in bucket: " << umap.bucket(test_key) << std::endl;

    // Section 6: Swap functionality
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

    // Section 7: Custom hash and comparison function
    std::cout << "\n=== Custom Hash and Equality Function ===" << std::endl;
    struct CustomHash {
        size_t operator()(const std::string& key) const {
            // Simple custom hash: sum of ASCII values
            size_t hash = 0;
            for (char c : key) {
                hash += static_cast<size_t>(c);
            }
            return hash;
        }
    };
    
    struct CustomEqual {
        bool operator()(const std::string& lhs, const std::string& rhs) const {
            // Case-insensitive equality
            if (lhs.size() != rhs.size()) return false;
            for (size_t i = 0; i < lhs.size(); ++i) {
                if (std::tolower(lhs[i]) != std::tolower(rhs[i])) return false;
            }
            return true;
        }
    };
    
    UnorderedMap<std::string, int, CustomHash, CustomEqual> custom_umap;
    custom_umap["test"] = 1;
    custom_umap["TEST"] = 2;  // Should update "test" due to case-insensitive equality
    custom_umap["another"] = 3;
    
    std::cout << "Custom unordered map size: " << custom_umap.size() << std::endl;
    std::cout << "custom_umap[\"test\"]: " << custom_umap["test"] << std::endl;
    std::cout << "Custom unordered map contents:" << std::endl;
    for (const auto& pair : custom_umap) {
        std::cout << "  " << pair.first << " : " << pair.second << std::endl;
    }

    // Section 8: Edge cases
    std::cout << "\n=== Edge Cases ===" << std::endl;
    // Empty map
    UnorderedMap<int, std::string> empty_map;
    std::cout << "Empty map size: " << empty_map.size() << std::endl;
    std::cout << "Empty map begin() == end(): " << (empty_map.begin() == empty_map.end() ? "true" : "false") << std::endl;
    
    // Access non-existent key with at() - should throw exception
    try {
        empty_map.at(10);
        std::cout << "This should not be printed!" << std::endl;
    } catch (const std::out_of_range& ex) {
        std::cout << "Caught expected exception: " << ex.what() << std::endl;
    }
    
    // Access non-existent key with operator[] - should insert default value
    std::cout << "empty_map[10]: " << empty_map[10] << std::endl;
    std::cout << "After using operator[], size: " << empty_map.size() << std::endl;

    // Section 9: Test hash_function and key_eq accessors
    std::cout << "\n=== Hash Function and Key Equality ===" << std::endl;
    auto hash_func = custom_umap.hash_function();
    auto key_eq_func = custom_umap.key_eq();
    
    std::string test_str1 = "example";
    std::string test_str2 = "EXAMPLE";
    
    std::cout << "Hash of \"" << test_str1 << "\": " << hash_func(test_str1) << std::endl;
    std::cout << "Hash of \"" << test_str2 << "\": " << hash_func(test_str2) << std::endl;
    std::cout << "Are \"" << test_str1 << "\" and \"" << test_str2 
              << "\" equal: " << (key_eq_func(test_str1, test_str2) ? "true" : "false") << std::endl;

    // Final section: clear everything
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