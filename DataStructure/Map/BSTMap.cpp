/**
 * @file BSTMap.cpp
 * @brief Implementation of BSTMap methods.
 */

#include "BSTMap.hpp"

#include <string>

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::FindNode(const Node* Nd, const K& key) const
    -> const Node* {
    if (!Nd) return nullptr;
    if (!Comp(key, Nd->data.first) && !Comp(Nd->data.first, key))
        return Nd;
    if (Comp(key, Nd->data.first))
        return FindNode(Nd->left.get(), key);
    else
        return FindNode(Nd->right.get(), key);
}

template <typename K, typename V, typename Compare>
bool BSTMap<K, V, Compare>::InsertImpl(
        std::unique_ptr<Node>& Nd, const K& key,
        const V& value, Node* Parent) {
    if (!Nd) {
        Nd = std::make_unique<Node>(key, value, Parent);
        size_++;
        return true;
    }
    if (!Comp(key, Nd->data.first) && !Comp(Nd->data.first, key)) {
        Nd->data.second = value;
        return false;
    }
    if (Comp(key, Nd->data.first))
        return InsertImpl(Nd->left, key, value, Nd.get());
    else
        return InsertImpl(Nd->right, key, value, Nd.get());
}

template <typename K, typename V, typename Compare>
bool BSTMap<K, V, Compare>::InsertImpl(
        std::unique_ptr<Node>& Nd, K&& key,
        V&& value, Node* Parent) {
    if (!Nd) {
        Nd = std::make_unique<Node>(std::move(key), std::move(value), Parent);
        size_++;
        return true;
    }
    if (!Comp(key, Nd->data.first) && !Comp(Nd->data.first, key)) {
        Nd->data.second = std::move(value);
        return false;
    }
    if (Comp(key, Nd->data.first))
        return InsertImpl(Nd->left, std::move(key), std::move(value), Nd.get());
    else
        return InsertImpl(Nd->right, std::move(key), std::move(value), Nd.get());
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::InsertOrAssignImpl(
        std::unique_ptr<Node>& Nd, const K& key,
        const V& value, Node* Parent)
    -> std::pair<Node*, bool> {
    if (!Nd) {
        Nd = std::make_unique<Node>(key, value, Parent);
        size_++;
        return {Nd.get(), true};
    }
    if (!Comp(key, Nd->data.first) && !Comp(Nd->data.first, key)) {
        Nd->data.second = value;
        return {Nd.get(), false};
    }
    if (Comp(key, Nd->data.first))
        return InsertOrAssignImpl(Nd->left, key, value, Nd.get());
    else
        return InsertOrAssignImpl(Nd->right, key, value, Nd.get());
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::InsertOrAssignImpl(
        std::unique_ptr<Node>& Nd, K&& key,
        V&& value, Node* Parent)
    -> std::pair<Node*, bool> {
    if (!Nd) {
        Nd = std::make_unique<Node>(std::move(key), std::move(value), Parent);
        size_++;
        return {Nd.get(), true};
    }
    if (!Comp(key, Nd->data.first) && !Comp(Nd->data.first, key)) {
        Nd->data.second = std::move(value);
        return {Nd.get(), false};
    }
    if (Comp(key, Nd->data.first))
        return InsertOrAssignImpl(Nd->left, std::move(key),
                                 std::move(value), Nd.get());
    else
        return InsertOrAssignImpl(Nd->right, std::move(key),
                                 std::move(value), Nd.get());
}

template <typename K, typename V, typename Compare>
bool BSTMap<K, V, Compare>::RemoveImpl(
        std::unique_ptr<Node>& Nd, const K& key) {
    if (!Nd) return false;

    if (Comp(key, Nd->data.first)) {
        return RemoveImpl(Nd->left, key);
    } else if (Comp(Nd->data.first, key)) {
        return RemoveImpl(Nd->right, key);
    } else {
        // Found the node to remove
        if (!Nd->left && !Nd->right) {
            Nd.reset();
            size_--;
            return true;
        } else if (!Nd->left) {
            Node* Parent = Nd->parent;
            Nd->right->parent = Parent;
            Nd = std::move(Nd->right);
            size_--;
            return true;
        } else if (!Nd->right) {
            Node* Parent = Nd->parent;
            Nd->left->parent = Parent;
            Nd = std::move(Nd->left);
            size_--;
            return true;
        } else {
            Node* Successor = FindMin(Nd->right.get());
            Nd->data = Successor->data;
            return RemoveImpl(Nd->right, Successor->data.first);
        }
    }
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::FindMin(Node* Nd) const -> Node* {
    if (!Nd) return nullptr;
    while (Nd->left) Nd = Nd->left.get();
    return Nd;
}

template <typename K, typename V, typename Compare>
void BSTMap<K, V, Compare>::InorderKeys(
        const std::unique_ptr<Node>& Nd, std::vector<K>& Result) const {
    if (Nd) {
        InorderKeys(Nd->left, Result);
        Result.push_back(Nd->data.first);
        InorderKeys(Nd->right, Result);
    }
}

template <typename K, typename V, typename Compare>
void BSTMap<K, V, Compare>::InorderValues(
        const std::unique_ptr<Node>& Nd, std::vector<V>& Result) const {
    if (Nd) {
        InorderValues(Nd->left, Result);
        Result.push_back(Nd->data.second);
        InorderValues(Nd->right, Result);
    }
}

template <typename K, typename V, typename Compare>
void BSTMap<K, V, Compare>::InorderEntries(
        const std::unique_ptr<Node>& Nd,
        std::vector<std::pair<K, V>>& Result) const {
    if (Nd) {
        InorderEntries(Nd->left, Result);
        Result.push_back(Nd->data);
        InorderEntries(Nd->right, Result);
    }
}

// ---------------------------------------------------------------------------
// Construction / assignment
// ---------------------------------------------------------------------------

template <typename K, typename V, typename Compare>
BSTMap<K, V, Compare>::BSTMap()
    : Root(nullptr), Comp{}, size_(0) {}

template <typename K, typename V, typename Compare>
BSTMap<K, V, Compare>::BSTMap(Compare Comparator)
    : Root(nullptr), Comp(std::move(Comparator)), size_(0) {}

template <typename K, typename V, typename Compare>
BSTMap<K, V, Compare>::BSTMap(std::initializer_list<std::pair<K, V>> Init)
    : Root(nullptr), Comp{}, size_(0) {
    std::lock_guard<std::mutex> Lock(mutex);
    for (const auto& pair : Init) {
        InsertImpl(Root, pair.first, pair.second, nullptr);
    }
}

template <typename K, typename V, typename Compare>
BSTMap<K, V, Compare>::BSTMap(BSTMap&& Other) noexcept
    : Comp(std::move(Other.Comp)), size_(Other.size_) {
    std::lock_guard<std::mutex> Lock(Other.mutex);
    Root = std::move(Other.Root);
    Other.size_ = 0;
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::operator=(BSTMap&& Other) noexcept
    -> BSTMap& {
    if (this != &Other) {
        std::lock_guard<std::mutex> Lock1(mutex);
        std::lock_guard<std::mutex> Lock2(Other.mutex);
        Root = std::move(Other.Root);
        Comp = std::move(Other.Comp);
        size_ = Other.size_;
        Other.size_ = 0;
    }
    return *this;
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::operator=(
        std::initializer_list<std::pair<K, V>> Ilist)
    -> BSTMap& {
    std::lock_guard<std::mutex> Lock(mutex);
    Root.reset();
    size_ = 0;
    for (const auto& pair : Ilist) {
        InsertImpl(Root, pair.first, pair.second, nullptr);
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Core API
// ---------------------------------------------------------------------------

template <typename K, typename V, typename Compare>
bool BSTMap<K, V, Compare>::insert(const K& key, const V& value) {
    std::lock_guard<std::mutex> Lock(mutex);
    return InsertImpl(Root, key, value, nullptr);
}

template <typename K, typename V, typename Compare>
bool BSTMap<K, V, Compare>::insert(K&& key, V&& value) {
    std::lock_guard<std::mutex> Lock(mutex);
    return InsertImpl(Root, std::move(key), std::move(value), nullptr);
}

template <typename K, typename V, typename Compare>
bool BSTMap<K, V, Compare>::insert(const std::pair<K, V>& pair) {
    return insert(pair.first, pair.second);
}

template <typename K, typename V, typename Compare>
bool BSTMap<K, V, Compare>::insert(std::pair<K, V>&& pair) {
    return insert(std::move(pair.first), std::move(pair.second));
}

template <typename K, typename V, typename Compare>
void BSTMap<K, V, Compare>::insert(
        std::initializer_list<std::pair<K, V>> Ilist) {
    std::lock_guard<std::mutex> Lock(mutex);
    for (const auto& pair : Ilist) {
        InsertImpl(Root, pair.first, pair.second, nullptr);
    }
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::InsertOrAssign(const K& key, const V& value)
    -> std::pair<iterator, bool> {
    std::lock_guard<std::mutex> Lock(mutex);
    auto Result = InsertOrAssignImpl(Root, key, value, nullptr);
    return {iterator(Result.first, this), Result.second};
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::InsertOrAssign(K&& key, V&& value)
    -> std::pair<iterator, bool> {
    std::lock_guard<std::mutex> Lock(mutex);
    auto Result = InsertOrAssignImpl(
        Root, std::move(key), std::move(value), nullptr);
    return {iterator(Result.first, this), Result.second};
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::find(const K& key) const
    -> std::optional<V> {
    std::lock_guard<std::mutex> Lock(mutex);
    const Node* Nd = FindNode(Root.get(), key);
    if (!Nd) return std::nullopt;
    return Nd->data.second;
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::FindIterator(const K& key)
    -> iterator {
    std::lock_guard<std::mutex> Lock(mutex);
    Node* Nd = const_cast<Node*>(FindNode(Root.get(), key));
    return Nd ? iterator(Nd, this) : end();
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::FindIterator(const K& key) const
    -> const_iterator {
    std::lock_guard<std::mutex> Lock(mutex);
    const Node* Nd = FindNode(Root.get(), key);
    return Nd ? const_iterator(Nd, this) : end();
}

template <typename K, typename V, typename Compare>
bool BSTMap<K, V, Compare>::Contains(const K& key) const {
    std::lock_guard<std::mutex> Lock(mutex);
    return FindNode(Root.get(), key) != nullptr;
}

template <typename K, typename V, typename Compare>
V& BSTMap<K, V, Compare>::operator[](const K& key) {
    std::lock_guard<std::mutex> Lock(mutex);
    auto Result = InsertOrAssignImpl(Root, key, V{}, nullptr);
    return Result.first->data.second;
}

template <typename K, typename V, typename Compare>
V& BSTMap<K, V, Compare>::operator[](K&& key) {
    std::lock_guard<std::mutex> Lock(mutex);
    auto Result = InsertOrAssignImpl(Root, std::move(key), V{}, nullptr);
    return Result.first->data.second;
}

// ---------------------------------------------------------------------------
// Bounds / range
// ---------------------------------------------------------------------------

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::lower_bound(const K& key)
    -> iterator {
    std::lock_guard<std::mutex> Lock(mutex);
    if (!Root) return end();
    Node* Result = nullptr;
    Node* Current = Root.get();
    while (Current) {
        if (!Comp(Current->data.first, key)) {
            Result = Current;
            Current = Current->left.get();
        } else {
            Current = Current->right.get();
        }
    }
    return Result ? iterator(Result, this) : end();
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::lower_bound(const K& key) const
    -> const_iterator {
    std::lock_guard<std::mutex> Lock(mutex);
    if (!Root) return end();
    const Node* Result = nullptr;
    const Node* Current = Root.get();
    while (Current) {
        if (!Comp(Current->data.first, key)) {
            Result = Current;
            Current = Current->left.get();
        } else {
            Current = Current->right.get();
        }
    }
    return Result ? const_iterator(Result, this) : end();
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::upper_bound(const K& key)
    -> iterator {
    std::lock_guard<std::mutex> Lock(mutex);
    if (!Root) return end();
    Node* Result = nullptr;
    Node* Current = Root.get();
    while (Current) {
        if (Comp(key, Current->data.first)) {
            Result = Current;
            Current = Current->left.get();
        } else {
            Current = Current->right.get();
        }
    }
    return Result ? iterator(Result, this) : end();
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::upper_bound(const K& key) const
    -> const_iterator {
    std::lock_guard<std::mutex> Lock(mutex);
    if (!Root) return end();
    const Node* Result = nullptr;
    const Node* Current = Root.get();
    while (Current) {
        if (Comp(key, Current->data.first)) {
            Result = Current;
            Current = Current->left.get();
        } else {
            Current = Current->right.get();
        }
    }
    return Result ? const_iterator(Result, this) : end();
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::equal_range(const K& key)
    -> std::pair<iterator, iterator> {
    return {lower_bound(key), upper_bound(key)};
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::equal_range(const K& key) const
    -> std::pair<const_iterator, const_iterator> {
    return {lower_bound(key), upper_bound(key)};
}

// ---------------------------------------------------------------------------
// Remove / erase
// ---------------------------------------------------------------------------

template <typename K, typename V, typename Compare>
bool BSTMap<K, V, Compare>::remove(const K& key) {
    std::lock_guard<std::mutex> Lock(mutex);
    return RemoveImpl(Root, key);
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::erase(iterator Pos) -> iterator {
    if (Pos == end()) return end();
    K KeyToRemove = Pos->first;
    iterator next = Pos;
    ++next;
    std::lock_guard<std::mutex> Lock(mutex);
    RemoveImpl(Root, KeyToRemove);
    return next;
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::erase(iterator first, iterator Last)
    -> iterator {
    if (first == Last) return Last;
    std::vector<K> KeysToRemove;
    for (auto It = first; It != Last; ++It) {
        KeysToRemove.push_back(It->first);
    }
    std::lock_guard<std::mutex> Lock(mutex);
    for (const auto& key : KeysToRemove) {
        RemoveImpl(Root, key);
    }
    return Last;
}

// ---------------------------------------------------------------------------
// Capacity
// ---------------------------------------------------------------------------

template <typename K, typename V, typename Compare>
std::size_t BSTMap<K, V, Compare>::size() const {
    std::lock_guard<std::mutex> Lock(mutex);
    return size_;
}

template <typename K, typename V, typename Compare>
bool BSTMap<K, V, Compare>::empty() const {
    std::lock_guard<std::mutex> Lock(mutex);
    return size_ == 0;
}

template <typename K, typename V, typename Compare>
void BSTMap<K, V, Compare>::clear() {
    std::lock_guard<std::mutex> Lock(mutex);
    Root.reset();
    size_ = 0;
}

// ---------------------------------------------------------------------------
// Bulk accessors
// ---------------------------------------------------------------------------

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::Keys() const -> std::vector<K> {
    std::lock_guard<std::mutex> Lock(mutex);
    std::vector<K> Result;
    Result.reserve(size_);
    InorderKeys(Root, Result);
    return Result;
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::Values() const -> std::vector<V> {
    std::lock_guard<std::mutex> Lock(mutex);
    std::vector<V> Result;
    Result.reserve(size_);
    InorderValues(Root, Result);
    return Result;
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::Entries() const
    -> std::vector<std::pair<K, V>> {
    std::lock_guard<std::mutex> Lock(mutex);
    std::vector<std::pair<K, V>> Result;
    Result.reserve(size_);
    InorderEntries(Root, Result);
    return Result;
}

template <typename K, typename V, typename Compare>
void BSTMap<K, V, Compare>::swap(BSTMap& Other) noexcept {
    if (this != &Other) {
        std::lock_guard<std::mutex> Lock1(mutex);
        std::lock_guard<std::mutex> Lock2(Other.mutex);
        std::swap(Root, Other.Root);
        std::swap(Comp, Other.Comp);
        std::swap(size_, Other.size_);
    }
}

// ---------------------------------------------------------------------------
// Iterators
// ---------------------------------------------------------------------------

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::begin() -> iterator {
    std::lock_guard<std::mutex> Lock(mutex);
    if (!Root) return end();
    Node* Leftmost = Root.get();
    while (Leftmost->left) Leftmost = Leftmost->left.get();
    return iterator(Leftmost, this);
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::end() -> iterator {
    return iterator(nullptr, this);
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::begin() const -> const_iterator {
    std::lock_guard<std::mutex> Lock(mutex);
    if (!Root) return end();
    const Node* Leftmost = Root.get();
    while (Leftmost->left) Leftmost = Leftmost->left.get();
    return const_iterator(Leftmost, this);
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::end() const -> const_iterator {
    return const_iterator(nullptr, this);
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::cbegin() const -> const_iterator {
    return begin();
}

template <typename K, typename V, typename Compare>
auto BSTMap<K, V, Compare>::cend() const -> const_iterator {
    return end();
}

// ---------------------------------------------------------------------------
// Explicit template instantiations
// ---------------------------------------------------------------------------

template class BSTMap<int, int>;
template class BSTMap<int, std::string>;
template class BSTMap<std::string, int>;
template class BSTMap<std::string, std::string>;
