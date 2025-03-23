#include <iostream>
#include <vector>
#include <list>
#include <functional>
#include <utility>
#include <stdexcept>
#include <iterator>
#include <limits>
#include <algorithm>

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
    T & operator[](const Key & key) {
        iterator it = find(key);
        if (it == end()) {
            std::pair<const Key, T> new_val(key, T());
            size_t idx = bucket_index(key);
            buckets[idx].push_back(new_val);
            ++num_elements;
            iterator ret(this, idx, --buckets[idx].end());
            return ret->second;
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
        std::pair<const Key, T> new_val(std::forward<Args>(args)...);
        size_t idx = bucket_index(new_val.first);
        typename std::list< std::pair<const Key, T> >::iterator it = buckets[idx].begin();
        for (; it != buckets[idx].end(); ++it) {
            if (key_equal_obj(it->first, new_val.first))
                return std::pair<iterator, bool>(iterator(this, idx, it), false);
        }
        buckets[idx].push_back(new_val);
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



int main() {
    // Section 1: Basic operations
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
