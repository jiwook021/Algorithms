/**
 * @file deque.hpp
 * @brief Production-grade doubly-linked-list deque.
 *
 * A node-based, double-ended queue that supports O(1) push/pop at both
 * ends with bidirectional iteration.
 *
 * Use cases:
 *   - Sliding window algorithms, undo/redo buffers, work-stealing queues.
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <initializer_list>
#include <algorithm>

// =============================================================================
// Deque — doubly-linked-list double-ended queue
// =============================================================================

/**
 * @class Deque
 * @tparam T Element type.
 *
 * Invariant: head_ and tail_ are nullptr iff Size_ == 0.
 * Each node links to its predecessor and successor, enabling O(1) insertion
 * and deletion at both ends.
 *
 * Complexity:
 *   PushFront / PushBack / PopFront / PopBack  — O(1)
 *   size / empty                                     — O(1)
 *   front / back                                     — O(1)
 *   clear                                            — O(n)
 * Memory: O(n) — one heap allocation per element.
 */
template <typename T>
class Deque {
private:
    struct Node {
        T      Data;
        Node*  Prev;
        Node*  Next;

        template <typename... Args>
        explicit Node(Args&&... args)
            : Data(std::forward<Args>(args)...), Prev(nullptr), Next(nullptr) {}
    };

    Node*       Head_;
    Node*       Tail_;
    std::size_t Size_;

public:
    // ── Bidirectional iterator ────────────────────────────────────────────

    class iterator {
        Node* Node_;
        // 'friend' lets the Deque class call the private constructor below.
        // Without it, Deque::begin()/end() couldn't create iterator objects
        // because the constructor is private (we don't want outside code
        // creating iterators from raw Node pointers).
        friend class Deque;
        explicit iterator(Node* n) noexcept : Node_(n) {}
    public:
        T& operator*()  const noexcept { return Node_->Data; }
        T* operator->() const noexcept { return &Node_->Data; }

        iterator& operator++()    noexcept { Node_ = Node_->Next; return *this; }
        iterator  operator++(int) noexcept { auto t = *this; Node_ = Node_->Next; return t; }
        iterator& operator--()    noexcept { Node_ = Node_->Prev; return *this; }
        iterator  operator--(int) noexcept { auto t = *this; Node_ = Node_->Prev; return t; }

        bool operator==(const iterator& o) const noexcept { return Node_ == o.Node_; }
        bool operator!=(const iterator& o) const noexcept { return Node_ != o.Node_; }
    };

    class const_iterator {
        const Node* Node_;
        // Same reason as iterator above: Deque needs access to the private
        // constructor to create const_iterators in begin()/end() const.
        friend class Deque;
        explicit const_iterator(const Node* n) noexcept : Node_(n) {}
    public:
        const T& operator*()  const noexcept { return Node_->Data; }
        const T* operator->() const noexcept { return &Node_->Data; }

        const_iterator& operator++()    noexcept { Node_ = Node_->Next; return *this; }
        const_iterator  operator++(int) noexcept { auto t = *this; Node_ = Node_->Next; return t; }
        const_iterator& operator--()    noexcept { Node_ = Node_->Prev; return *this; }
        const_iterator  operator--(int) noexcept { auto t = *this; Node_ = Node_->Prev; return t; }

        bool operator==(const const_iterator& o) const noexcept { return Node_ == o.Node_; }
        bool operator!=(const const_iterator& o) const noexcept { return Node_ != o.Node_; }
    };

    // ── Construction / destruction ────────────────────────────────────────

    /** Default-construct an empty deque. */
    Deque() noexcept : Head_(nullptr), Tail_(nullptr), Size_(0) {}

    /** Construct from initializer list. */
    Deque(std::initializer_list<T> Init) : Deque() {
        for (const auto& v : Init)
            PushBack(v);
    }

    /** Copy constructor — deep copy of all nodes. */
    Deque(const Deque& Other) : Deque() {
        for (const Node* n = Other.Head_; n; n = n->Next)
            PushBack(n->Data);
    }

    /** Move constructor — steals nodes from @p other. */
    Deque(Deque&& Other) noexcept
        : Head_(Other.Head_), Tail_(Other.Tail_), Size_(Other.Size_) {
        Other.Head_ = Other.Tail_ = nullptr;
        Other.Size_ = 0;
    }

    /** Copy assignment (copy-and-swap). */
    Deque& operator=(const Deque& Other) {
        if (this != &Other) {
            Deque Tmp(Other);
            Swap(Tmp);
        }
        return *this;
    }

    /** Move assignment. */
    Deque& operator=(Deque&& Other) noexcept {
        if (this != &Other) {
            Clear();
            Head_  = Other.Head_;
            Tail_  = Other.Tail_;
            Size_  = Other.Size_;
            Other.Head_ = Other.Tail_ = nullptr;
            Other.Size_ = 0;
        }
        return *this;
    }

    /** Destructor — frees all nodes. */
    ~Deque() noexcept { Clear(); }

    // ── Iterators ─────────────────────────────────────────────────────────

    iterator       begin()        noexcept { return iterator(Head_); }
    iterator       end()          noexcept { return iterator(nullptr); }
    const_iterator begin()  const noexcept { return const_iterator(Head_); }
    const_iterator end()    const noexcept { return const_iterator(nullptr); }
    const_iterator cbegin() const noexcept { return begin(); }
    const_iterator cend()   const noexcept { return end(); }

    // ── Capacity ──────────────────────────────────────────────────────────

    [[nodiscard]] std::size_t Size()  const noexcept { return Size_; }
    [[nodiscard]] bool        Empty() const noexcept { return Size_ == 0; }

    // ── Element access ────────────────────────────────────────────────────

    T&       Front()       { CheckEmpty("front"); return Head_->Data; }
    const T& Front() const { CheckEmpty("front"); return Head_->Data; }
    T&       Back()        { CheckEmpty("back");  return Tail_->Data; }
    const T& Back()  const { CheckEmpty("back");  return Tail_->Data; }

    // ── Modifiers ─────────────────────────────────────────────────────────

    /** Insert @p value at the front. O(1). */
    void PushFront(const T& value) { EmplaceFront(value); }
    void PushFront(T&& value)      { EmplaceFront(std::move(value)); }

    /** Insert @p value at the back. O(1). */
    void PushBack(const T& value) { EmplaceBack(value); }
    void PushBack(T&& value)      { EmplaceBack(std::move(value)); }

    /** Construct in-place at front. */
    template <typename... Args>
    void EmplaceFront(Args&&... args) {
        Node* n = new Node(std::forward<Args>(args)...);
        n->Next = Head_;
        if (Head_) Head_->Prev = n;
        else       Tail_ = n;
        Head_ = n;
        ++Size_;
    }

    /** Construct in-place at back. */
    template <typename... Args>
    void EmplaceBack(Args&&... args) {
        Node* n = new Node(std::forward<Args>(args)...);
        n->Prev = Tail_;
        if (Tail_) Tail_->Next = n;
        else       Head_ = n;
        Tail_ = n;
        ++Size_;
    }

    /** Remove the front element. @throws std::underflow_error */
    void PopFront() {
        CheckEmpty("PopFront");
        Node* Old = Head_;
        Head_ = Head_->Next;
        if (Head_) Head_->Prev = nullptr;
        else       Tail_ = nullptr;
        delete Old;
        --Size_;
    }

    /** Remove the back element. @throws std::underflow_error */
    void PopBack() {
        CheckEmpty("PopBack");
        Node* Old = Tail_;
        Tail_ = Tail_->Prev;
        if (Tail_) Tail_->Next = nullptr;
        else       Head_ = nullptr;
        delete Old;
        --Size_;
    }

    /** Remove all elements. O(n). */
    void Clear() noexcept {
        while (Head_) {
            Node* Next = Head_->Next;
            delete Head_;
            Head_ = Next;
        }
        Tail_ = nullptr;
        Size_ = 0;
    }

    /** Swap contents with another deque. O(1). */
    void Swap(Deque& Other) noexcept {
        std::swap(Head_, Other.Head_);
        std::swap(Tail_, Other.Tail_);
        std::swap(Size_, Other.Size_);
    }

private:
    void CheckEmpty(const char* Fn) const {
        if (Empty())
            throw std::underflow_error(std::string("Deque::") + Fn + "(): deque is empty");
    }
};
