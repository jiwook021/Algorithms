/**
 * @file SelfOrganizingList.cpp
 * @brief Manual-priority SelfOrganizingList implementation.
 */

#include "SelfOrganizingList.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace {
constexpr const char* kEmptyListMessage = "SelfOrganizingList is empty";
}

template <typename T>
SelfOrganizingList<T>::Node::Node(
    const T& Item,
    int PriorityValue,
    std::size_t SequenceValue)
    : Data(Item),
      Priority(PriorityValue),
      Sequence(SequenceValue),
      Next(nullptr) {}

template <typename T>
SelfOrganizingList<T>::SelfOrganizingList()
    : Head(nullptr),
      NextSequence(0) {}

template <typename T>
SelfOrganizingList<T>::~SelfOrganizingList() {
    Clear();
}

template <typename T>
SelfOrganizingList<T>::SelfOrganizingList(SelfOrganizingList&& Other) noexcept
    : Head(Other.Head),
      NextSequence(Other.NextSequence) {
    Other.Head = nullptr;
    Other.NextSequence = 0;
}

template <typename T>
SelfOrganizingList<T>& SelfOrganizingList<T>::operator=(
    SelfOrganizingList&& Other) noexcept {
    if (this != &Other) {
        Clear();
        Head = Other.Head;
        NextSequence = Other.NextSequence;
        Other.Head = nullptr;
        Other.NextSequence = 0;
    }

    return *this;
}

template <typename T>
void SelfOrganizingList<T>::Insert(const T& Item, int Priority) {
    InsertNode(new Node(Item, Priority, NextSequence++));
}

template <typename T>
T SelfOrganizingList<T>::Pop() {
    if (Empty()) {
        throw std::out_of_range(kEmptyListMessage);
    }

    Node* Front = Head;
    Head = Head->Next;

    T Result = std::move(Front->Data);
    delete Front;
    return Result;
}

template <typename T>
bool SelfOrganizingList<T>::UpdatePriority(const T& Item, int Priority) {
    Node* Previous = nullptr;
    Node* Current = Head;

    while (Current) {
        if (Current->Data == Item) {
            if (Previous) {
                Previous->Next = Current->Next;
            } else {
                Head = Current->Next;
            }

            Current->Priority = Priority;
            Current->Next = nullptr;
            InsertNode(Current);
            return true;
        }

        Previous = Current;
        Current = Current->Next;
    }

    return false;
}

template <typename T>
bool SelfOrganizingList<T>::Find(const T& Item) const {
    const Node* Current = Head;

    while (Current) {
        if (Current->Data == Item) {
            return true;
        }

        Current = Current->Next;
    }

    return false;
}

template <typename T>
std::vector<T> SelfOrganizingList<T>::ToVector() const {
    std::vector<T> Result;
    const Node* Current = Head;

    while (Current) {
        Result.push_back(Current->Data);
        Current = Current->Next;
    }

    return Result;
}

template <typename T>
std::vector<std::pair<T, int>> SelfOrganizingList<T>::ToPriorityVector() const {
    std::vector<std::pair<T, int>> Result;
    const Node* Current = Head;

    while (Current) {
        Result.emplace_back(Current->Data, Current->Priority);
        Current = Current->Next;
    }

    return Result;
}

template <typename T>
bool SelfOrganizingList<T>::Empty() const noexcept {
    return Head == nullptr;
}

template <typename T>
std::size_t SelfOrganizingList<T>::Size() const noexcept {
    std::size_t Result = 0;
    const Node* Current = Head;

    while (Current) {
        ++Result;
        Current = Current->Next;
    }

    return Result;
}

template <typename T>
void SelfOrganizingList<T>::Clear() noexcept {
    while (Head) {
        Node* Temp = Head;
        Head = Head->Next;
        delete Temp;
    }

    NextSequence = 0;
}

template <typename T>
void SelfOrganizingList<T>::InsertNode(Node* NewNode) {
    if (!Head || ShouldComeBefore(NewNode, Head)) {
        NewNode->Next = Head;
        Head = NewNode;
        return;
    }

    Node* Current = Head;
    while (Current->Next && !ShouldComeBefore(NewNode, Current->Next)) {
        Current = Current->Next;
    }

    NewNode->Next = Current->Next;
    Current->Next = NewNode;
}

template <typename T>
bool SelfOrganizingList<T>::ShouldComeBefore(
    const Node* Candidate,
    const Node* Current) {
    if (Candidate->Priority != Current->Priority) {
        return Candidate->Priority > Current->Priority;
    }

    return Candidate->Sequence < Current->Sequence;
}

template class SelfOrganizingList<int>;
template class SelfOrganizingList<std::string>;
