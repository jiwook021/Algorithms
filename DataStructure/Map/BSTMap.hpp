/**
 * @file BSTMap.hpp
 * @brief Binary search tree based ordered map (declarations).
 *
 * @details
 * An ordered map stores key-value pairs sorted by key and provides
 * O(log n) average-time insert, find, and remove.  In-order iteration
 * visits keys in ascending order — the defining property of a BST.
 *
 * Complexity:
 *   Insert / Find / Remove : O(log n) average, O(n) worst case
 *   In-order traversal     : O(n)
 *   Space                  : O(n)
 */

#pragma once

#include "BSTNode.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

/**
 * @class BSTMap
 * @brief Ordered map backed by a binary search tree.
 *
 * @tparam K       Key type (must support < comparison).
 * @tparam V       Value type.
 * @tparam Compare Comparison functor (default: std::less<K>).
 */
template <typename K, typename V, typename Compare = std::less<K>>
class BSTMap {
private:
    using Node = BSTNode<K, V>;

    std::unique_ptr<Node> Root;
    Compare Comp;
    mutable std::mutex mutex;
    std::size_t size_;

    // Private helpers (implemented in BSTMap.cpp)
    const Node* FindNode(const Node* Nd, const K& key) const;
    bool InsertImpl(std::unique_ptr<Node>& Nd, const K& key,
                    const V& value, Node* Parent);
    bool InsertImpl(std::unique_ptr<Node>& Nd, K&& key,
                    V&& value, Node* Parent);
    std::pair<Node*, bool> InsertOrAssignImpl(
        std::unique_ptr<Node>& Nd, const K& key,
        const V& value, Node* Parent);
    std::pair<Node*, bool> InsertOrAssignImpl(
        std::unique_ptr<Node>& Nd, K&& key,
        V&& value, Node* Parent);
    bool RemoveImpl(std::unique_ptr<Node>& Nd, const K& key);
    Node* FindMin(Node* Nd) const;
    void InorderKeys(const std::unique_ptr<Node>& Nd,
                     std::vector<K>& Result) const;
    void InorderValues(const std::unique_ptr<Node>& Nd,
                       std::vector<V>& Result) const;
    void InorderEntries(const std::unique_ptr<Node>& Nd,
                        std::vector<std::pair<K, V>>& Result) const;

    /** Must stay inline (member function template). */
    template <typename Func>
    void InorderApply(const std::unique_ptr<Node>& Nd, Func&& Fn) const {
        if (Nd) {
            InorderApply(Nd->left, Fn);
            Fn(Nd->data.first, Nd->data.second);
            InorderApply(Nd->right, Fn);
        }
    }

public:
    class iterator;
    class const_iterator;

    // ---------------------------------------------------------------
    // Iterator (inline — tightly coupled to tree internals)
    // ---------------------------------------------------------------

    class iterator {
    private:
        Node* Current;
        const BSTMap* map;
        friend class const_iterator;
        friend class BSTMap;

    public:
        iterator() : Current(nullptr), map(nullptr) {}

        iterator(Node* nd, const BSTMap* m) : Current(nd), map(m) {}

        std::pair<K, V>& operator*() const {
            return Current->data;
        }

        std::pair<K, V>* operator->() const {
            return &(Current->data);
        }

        iterator& operator++() {
            if (!Current) return *this;
            if (Current->right) {
                Current = Current->right.get();
                while (Current->left) Current = Current->left.get();
            } else {
                Node* Parent = Current->parent;
                while (Parent && Current == Parent->right.get()) {
                    Current = Parent;
                    Parent = Parent->parent;
                }
                Current = Parent;
            }
            return *this;
        }

        iterator operator++(int) {
            iterator Temp = *this;
            ++(*this);
            return Temp;
        }

        iterator& operator--() {
            if (!Current) {
                if (map && map->Root) {
                    Current = map->Root.get();
                    while (Current->right) Current = Current->right.get();
                }
                return *this;
            }
            if (Current->left) {
                Current = Current->left.get();
                while (Current->right) Current = Current->right.get();
            } else {
                Node* Parent = Current->parent;
                while (Parent && Current == Parent->left.get()) {
                    Current = Parent;
                    Parent = Parent->parent;
                }
                Current = Parent;
            }
            return *this;
        }

        iterator operator--(int) {
            iterator Temp = *this;
            --(*this);
            return Temp;
        }

        bool operator==(const iterator& Other) const {
            return Current == Other.Current;
        }

        bool operator!=(const iterator& Other) const {
            return Current != Other.Current;
        }
    };

    // ---------------------------------------------------------------
    // Const iterator (inline)
    // ---------------------------------------------------------------

    class const_iterator {
    private:
        const Node* Current;
        const BSTMap* map;

    public:
        const_iterator() : Current(nullptr), map(nullptr) {}

        const_iterator(const Node* nd, const BSTMap* m)
            : Current(nd), map(m) {}

        const_iterator(const iterator& It)
            : Current(It.Current), map(It.map) {}

        const std::pair<K, V>& operator*() const {
            return Current->data;
        }

        const std::pair<K, V>* operator->() const {
            return &(Current->data);
        }

        const_iterator& operator++() {
            if (!Current) return *this;
            if (Current->right) {
                Current = Current->right.get();
                while (Current->left) Current = Current->left.get();
            } else {
                const Node* Parent = Current->parent;
                while (Parent && Current == Parent->right.get()) {
                    Current = Parent;
                    Parent = Parent->parent;
                }
                Current = Parent;
            }
            return *this;
        }

        const_iterator operator++(int) {
            const_iterator Temp = *this;
            ++(*this);
            return Temp;
        }

        const_iterator& operator--() {
            if (!Current) {
                if (map && map->Root) {
                    Current = map->Root.get();
                    while (Current->right) Current = Current->right.get();
                }
                return *this;
            }
            if (Current->left) {
                Current = Current->left.get();
                while (Current->right) Current = Current->right.get();
            } else {
                const Node* Parent = Current->parent;
                while (Parent && Current == Parent->left.get()) {
                    Current = Parent;
                    Parent = Parent->parent;
                }
                Current = Parent;
            }
            return *this;
        }

        const_iterator operator--(int) {
            const_iterator Temp = *this;
            --(*this);
            return Temp;
        }

        bool operator==(const const_iterator& Other) const {
            return Current == Other.Current;
        }

        bool operator!=(const const_iterator& Other) const {
            return Current != Other.Current;
        }
    };

    // ---------------------------------------------------------------
    // Construction / assignment
    // ---------------------------------------------------------------

    BSTMap();
    explicit BSTMap(Compare Comparator);
    BSTMap(std::initializer_list<std::pair<K, V>> Init);

    /** Must stay inline (constructor template). */
    template <typename InputIt>
    BSTMap(InputIt first, InputIt Last)
        : Root(nullptr), Comp{}, size_(0) {
        std::lock_guard<std::mutex> Lock(mutex);
        for (auto It = first; It != Last; ++It) {
            InsertImpl(Root, It->first, It->second, nullptr);
        }
    }

    ~BSTMap() = default;

    BSTMap(BSTMap&& Other) noexcept;
    BSTMap& operator=(BSTMap&& Other) noexcept;
    BSTMap& operator=(std::initializer_list<std::pair<K, V>> Ilist);

    BSTMap(const BSTMap&) = delete;
    BSTMap& operator=(const BSTMap&) = delete;

    // ---------------------------------------------------------------
    // Core API
    // ---------------------------------------------------------------

    bool insert(const K& key, const V& value);
    bool insert(K&& key, V&& value);
    bool insert(const std::pair<K, V>& pair);
    bool insert(std::pair<K, V>&& pair);
    void insert(std::initializer_list<std::pair<K, V>> Ilist);

    /** Must stay inline (member function template). */
    template <typename InputIt>
    void insert(InputIt first, InputIt Last) {
        std::lock_guard<std::mutex> Lock(mutex);
        for (auto It = first; It != Last; ++It) {
            InsertImpl(Root, It->first, It->second, nullptr);
        }
    }

    std::pair<iterator, bool> InsertOrAssign(const K& key, const V& value);
    std::pair<iterator, bool> InsertOrAssign(K&& key, V&& value);

    /** Must stay inline (variadic template). */
    template <typename... Args>
    std::pair<iterator, bool> Emplace(const K& key, Args&&... args) {
        std::lock_guard<std::mutex> Lock(mutex);
        std::unique_ptr<Node>* Current = &Root;
        Node* Parent = nullptr;
        while (*Current) {
            Parent = Current->get();
            if (!Comp(key, (*Current)->data.first) &&
                !Comp((*Current)->data.first, key)) {
                (*Current)->data.second = V(std::forward<Args>(args)...);
                return {iterator(Current->get(), this), false};
            }
            if (Comp(key, (*Current)->data.first))
                Current = &((*Current)->left);
            else
                Current = &((*Current)->right);
        }
        *Current = std::make_unique<Node>(
            key, V(std::forward<Args>(args)...), Parent);
        size_++;
        return {iterator(Current->get(), this), true};
    }

    /** Must stay inline (variadic template). */
    template <typename... Args>
    std::pair<iterator, bool> Emplace(K&& key, Args&&... args) {
        std::lock_guard<std::mutex> Lock(mutex);
        std::unique_ptr<Node>* Current = &Root;
        Node* Parent = nullptr;
        while (*Current) {
            Parent = Current->get();
            if (!Comp(key, (*Current)->data.first) &&
                !Comp((*Current)->data.first, key)) {
                (*Current)->data.second = V(std::forward<Args>(args)...);
                return {iterator(Current->get(), this), false};
            }
            if (Comp(key, (*Current)->data.first))
                Current = &((*Current)->left);
            else
                Current = &((*Current)->right);
        }
        *Current = std::make_unique<Node>(
            std::move(key), V(std::forward<Args>(args)...), Parent);
        size_++;
        return {iterator(Current->get(), this), true};
    }

    std::optional<V> find(const K& key) const;
    iterator FindIterator(const K& key);
    const_iterator FindIterator(const K& key) const;
    bool Contains(const K& key) const;

    V& operator[](const K& key);
    V& operator[](K&& key);

    iterator lower_bound(const K& key);
    const_iterator lower_bound(const K& key) const;
    iterator upper_bound(const K& key);
    const_iterator upper_bound(const K& key) const;
    std::pair<iterator, iterator> equal_range(const K& key);
    std::pair<const_iterator, const_iterator> equal_range(const K& key) const;

    bool remove(const K& key);
    iterator erase(iterator Pos);
    iterator erase(iterator first, iterator Last);

    // ---------------------------------------------------------------
    // Capacity
    // ---------------------------------------------------------------

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] bool empty() const;
    void clear();

    // ---------------------------------------------------------------
    // Bulk accessors
    // ---------------------------------------------------------------

    std::vector<K> Keys() const;
    std::vector<V> Values() const;
    std::vector<std::pair<K, V>> Entries() const;

    /** Must stay inline (member function template). */
    template <typename Func>
    void for_each(Func&& Fn) const {
        std::lock_guard<std::mutex> Lock(mutex);
        InorderApply(Root, std::forward<Func>(Fn));
    }

    void swap(BSTMap& Other) noexcept;

    // ---------------------------------------------------------------
    // Iterators
    // ---------------------------------------------------------------

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    const_iterator cbegin() const;
    const_iterator cend() const;
};

// ---------------------------------------------------------------------------
// Free comparison operators (kept inline — template functions)
// ---------------------------------------------------------------------------

template <typename K, typename V, typename Compare>
bool operator==(const BSTMap<K, V, Compare>& Lhs,
                const BSTMap<K, V, Compare>& Rhs) {
    if (Lhs.size() != Rhs.size()) return false;
    auto Lit = Lhs.begin();
    auto Rit = Rhs.begin();
    while (Lit != Lhs.end() && Rit != Rhs.end()) {
        if (Lit->first != Rit->first || Lit->second != Rit->second)
            return false;
        ++Lit;
        ++Rit;
    }
    return true;
}

template <typename K, typename V, typename Compare>
bool operator!=(const BSTMap<K, V, Compare>& Lhs,
                const BSTMap<K, V, Compare>& Rhs) {
    return !(Lhs == Rhs);
}

template <typename K, typename V, typename Compare>
bool operator<(const BSTMap<K, V, Compare>& Lhs,
               const BSTMap<K, V, Compare>& Rhs) {
    if (Lhs.size() != Rhs.size()) return Lhs.size() < Rhs.size();
    auto Lit = Lhs.begin();
    auto Rit = Rhs.begin();
    while (Lit != Lhs.end() && Rit != Rhs.end()) {
        if (Lit->first != Rit->first) return Lit->first < Rit->first;
        if (Lit->second != Rit->second) {
            std::string Lval = Lit->second;
            std::string Rval = Rit->second;
            std::transform(Lval.begin(), Lval.end(), Lval.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            std::transform(Rval.begin(), Rval.end(), Rval.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if (Lval == Rval)
                return !std::isupper(Lit->second[0]) &&
                       std::isupper(Rit->second[0]);
            return Lval < Rval;
        }
        ++Lit;
        ++Rit;
    }
    return false;
}

template <typename K, typename V, typename Compare>
bool operator>(const BSTMap<K, V, Compare>& Lhs,
               const BSTMap<K, V, Compare>& Rhs) {
    return Rhs < Lhs;
}

template <typename K, typename V, typename Compare>
bool operator<=(const BSTMap<K, V, Compare>& Lhs,
                const BSTMap<K, V, Compare>& Rhs) {
    return !(Rhs < Lhs);
}

template <typename K, typename V, typename Compare>
bool operator>=(const BSTMap<K, V, Compare>& Lhs,
                const BSTMap<K, V, Compare>& Rhs) {
    return !(Lhs < Rhs);
}

template <typename K, typename V, typename Compare>
void swap(BSTMap<K, V, Compare>& Lhs,
          BSTMap<K, V, Compare>& Rhs) noexcept {
    Lhs.swap(Rhs);
}
