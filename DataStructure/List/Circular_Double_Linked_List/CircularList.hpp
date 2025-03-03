/**
 * @file circular_list.hpp
 * @brief Production-grade circular doubly-linked list.
 *
 * A sentinel-node circular doubly-linked list that supports O(1) insertion
 * and removal at both ends, bidirectional iteration, and search.
 *
 * The sentinel (dummy) node simplifies edge-case handling: the list is
 * never truly "empty" at the node level — head always points to the
 * sentinel whose next/prev loop back to itself when the list is logically empty.
 *
 * Use cases:
 *   - LRU caches, round-robin schedulers, circular buffers.
 */

#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <algorithm>

// =============================================================================
// CircularList — sentinel-based circular doubly-linked list
// =============================================================================

/**
 * @class CircularList
 * @tparam T Element type.
 *
 * Invariant: sentinel_.next == sentinel_.prev == &sentinel_ iff size_ == 0.
 *
 * Complexity:
 *   push_front / push_back / pop_front / pop_back  — O(1)
 *   insert (at iterator) / erase (at iterator)     — O(1)
 *   find                                           — O(n)
 *   size / empty                                   — O(1)
 * Memory: O(n) — one heap allocation per element + 1 sentinel.
 */
template <typename T>
class CircularList {
private:
    struct Node {
        T     data;
        Node* Prev;
        Node* Next;

        template <typename... Args>
        explicit Node(Args&&... args)
            : data(std::forward<Args>(args)...), Prev(nullptr), Next(nullptr) {}

        // Sentinel constructor (data is default-initialised).
        Node() : data{}, Prev(this), Next(this) {}
    };

    Node        Sentinel_;
    std::size_t size_;

    /// Link @p NewNode between @p Before and @p After.
    void Link(Node* Before, Node* NewNode, Node* After) noexcept {
        Before->Next = NewNode;
        NewNode->Prev   = Before;
        NewNode->Next   = After;
        After->Prev  = NewNode;
    }

    /// Unlink @p Target from the chain and delete it.
    void Unlink(Node* Target) noexcept {
        Target->Prev->Next = Target->Next;
        Target->Next->Prev = Target->Prev;
        delete Target;
    }

public:
    // ── Bidirectional iterator ────────────────────────────────────────────

    class iterator {
        Node* Node_;
        friend class CircularList;
        explicit iterator(Node* n) noexcept : Node_(n) {}
    public:
        T& operator*()  const noexcept { return Node_->data; }
        T* operator->() const noexcept { return &Node_->data; }

        iterator& operator++()    noexcept { Node_ = Node_->Next; return *this; }
        iterator  operator++(int) noexcept { auto t = *this; ++(*this); return t; }
        iterator& operator--()    noexcept { Node_ = Node_->Prev; return *this; }
        iterator  operator--(int) noexcept { auto t = *this; --(*this); return t; }

        bool operator==(const iterator& o) const noexcept { return Node_ == o.Node_; }
        bool operator!=(const iterator& o) const noexcept { return Node_ != o.Node_; }
    };

    class const_iterator {
        const Node* Node_;
        friend class CircularList;
        explicit const_iterator(const Node* n) noexcept : Node_(n) {}
    public:
        const T& operator*()  const noexcept { return Node_->data; }
        const T* operator->() const noexcept { return &Node_->data; }

        const_iterator& operator++()    noexcept { Node_ = Node_->Next; return *this; }
        const_iterator  operator++(int) noexcept { auto t = *this; ++(*this); return t; }
        const_iterator& operator--()    noexcept { Node_ = Node_->Prev; return *this; }
        const_iterator  operator--(int) noexcept { auto t = *this; --(*this); return t; }

        bool operator==(const const_iterator& o) const noexcept { return Node_ == o.Node_; }
        bool operator!=(const const_iterator& o) const noexcept { return Node_ != o.Node_; }
    };

    // ── Construction / destruction ────────────────────────────────────────

    CircularList() noexcept : Sentinel_(), size_(0) {}

    CircularList(std::initializer_list<T> Init) : CircularList() {
        for (const auto& v : Init)
            push_back(v);
    }

    CircularList(const CircularList& Other) : CircularList() {
        for (auto It = Other.begin(); It != Other.end(); ++It)
            push_back(*It);
    }

    CircularList(CircularList&& Other) noexcept : CircularList() {
        if (!Other.Empty()) {
            Sentinel_.Next = Other.Sentinel_.Next;
            Sentinel_.Prev = Other.Sentinel_.Prev;
            Sentinel_.Next->Prev = &Sentinel_;
            Sentinel_.Prev->Next = &Sentinel_;
            size_ = Other.size_;
            Other.Sentinel_.Next = Other.Sentinel_.Prev = &Other.Sentinel_;
            Other.size_ = 0;
        }
    }

    CircularList& operator=(const CircularList& Other) {
        if (this != &Other) {
            CircularList Tmp(Other);
            swap(Tmp);
        }
        return *this;
    }

    CircularList& operator=(CircularList&& Other) noexcept {
        if (this != &Other) {
            Clear();
            if (!Other.Empty()) {
                Sentinel_.Next = Other.Sentinel_.Next;
                Sentinel_.Prev = Other.Sentinel_.Prev;
                Sentinel_.Next->Prev = &Sentinel_;
                Sentinel_.Prev->Next = &Sentinel_;
                size_ = Other.size_;
                Other.Sentinel_.Next = Other.Sentinel_.Prev = &Other.Sentinel_;
                Other.size_ = 0;
            }
        }
        return *this;
    }

    ~CircularList() noexcept { Clear(); }

    // ── Iterators ─────────────────────────────────────────────────────────

    iterator       begin()        noexcept { return iterator(Sentinel_.Next); }
    iterator       end()          noexcept { return iterator(&Sentinel_); }
    const_iterator begin()  const noexcept { return const_iterator(Sentinel_.Next); }
    const_iterator end()    const noexcept { return const_iterator(&Sentinel_); }
    const_iterator cbegin() const noexcept { return begin(); }
    const_iterator cend()   const noexcept { return end(); }

    // ── Capacity ──────────────────────────────────────────────────────────

    [[nodiscard]] std::size_t Size()  const noexcept { return size_; }
    [[nodiscard]] bool        Empty() const noexcept { return size_ == 0; }

    // ── Element access ────────────────────────────────────────────────────

    T&       Front()       { CheckEmpty("front"); return Sentinel_.Next->data; }
    const T& Front() const { CheckEmpty("front"); return Sentinel_.Next->data; }
    T&       Back()        { CheckEmpty("back");  return Sentinel_.Prev->data; }
    const T& Back()  const { CheckEmpty("back");  return Sentinel_.Prev->data; }

    // ── Modifiers ─────────────────────────────────────────────────────────

    void push_front(const T& v) { emplace_front(v); }
    void push_front(T&& v)      { emplace_front(std::move(v)); }
    void push_back(const T& v)  { emplace_back(v); }
    void push_back(T&& v)       { emplace_back(std::move(v)); }

    template <typename... Args>
    void emplace_front(Args&&... args) {
        Node* n = new Node(std::forward<Args>(args)...);
        Link(&Sentinel_, n, Sentinel_.Next);
        ++size_;
    }

    template <typename... Args>
    void emplace_back(Args&&... args) {
        Node* n = new Node(std::forward<Args>(args)...);
        Link(Sentinel_.Prev, n, &Sentinel_);
        ++size_;
    }

    /** Insert before @p pos. Returns iterator to new element. */
    iterator Insert(iterator Pos, const T& value) {
        Node* n = new Node(value);
        Link(Pos.Node_->Prev, n, Pos.Node_);
        ++size_;
        return iterator(n);
    }

    void pop_front() {
        CheckEmpty("pop_front");
        Unlink(Sentinel_.Next);
        --size_;
    }

    void pop_back() {
        CheckEmpty("pop_back");
        Unlink(Sentinel_.Prev);
        --size_;
    }

    /** Erase element at @p pos. Returns iterator to next element. */
    iterator Erase(iterator Pos) {
        Node* next = Pos.Node_->Next;
        Unlink(Pos.Node_);
        --size_;
        return iterator(next);
    }

    /** Remove first element equal to @p value. Returns true if found. */
    bool remove(const T& value) {
        for (Node* n = Sentinel_.Next; n != &Sentinel_; n = n->Next) {
            if (n->data == value) {
                Unlink(n);
                --size_;
                return true;
            }
        }
        return false;
    }

    /** Find first element equal to @p value. Returns end() if not found. */
    iterator Find(const T& value) {
        for (Node* n = Sentinel_.Next; n != &Sentinel_; n = n->Next)
            if (n->data == value) return iterator(n);
        return end();
    }

    void Clear() noexcept {
        Node* Cur = Sentinel_.Next;
        while (Cur != &Sentinel_) {
            Node* next = Cur->Next;
            delete Cur;
            Cur = next;
        }
        Sentinel_.Next = Sentinel_.Prev = &Sentinel_;
        size_ = 0;
    }

    void Swap(CircularList& Other) noexcept {
        if (this == &Other) return;

        // Save this's links
        Node* ThisFirst = Sentinel_.Next;
        Node* ThisLast  = Sentinel_.Prev;
        bool  ThisEmpty = Empty();

        // Save other's links
        Node* OtherFirst = Other.Sentinel_.Next;
        Node* OtherLast  = Other.Sentinel_.Prev;
        bool  OtherEmpty = Other.Empty();

        // Rewire this
        if (!OtherEmpty) {
            Sentinel_.Next = OtherFirst;
            Sentinel_.Prev = OtherLast;
            OtherFirst->Prev = &Sentinel_;
            OtherLast->Next  = &Sentinel_;
        } else {
            Sentinel_.Next = Sentinel_.Prev = &Sentinel_;
        }

        // Rewire other
        if (!ThisEmpty) {
            Other.Sentinel_.Next = ThisFirst;
            Other.Sentinel_.Prev = ThisLast;
            ThisFirst->Prev = &Other.Sentinel_;
            ThisLast->Next  = &Other.Sentinel_;
        } else {
            Other.Sentinel_.Next = Other.Sentinel_.Prev = &Other.Sentinel_;
        }

        std::swap(size_, Other.size_);
    }

private:
    void CheckEmpty(const char* Fn) const {
        if (Empty())
            throw std::underflow_error(
                std::string("CircularList::") + Fn + "(): list is empty");
    }
};
