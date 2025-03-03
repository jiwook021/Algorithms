/**
 * @file SelfOrganizingList.hpp
 * @brief Manual-priority self-organizing linked list declarations.
 * @details Stores items in descending priority order. Equal priorities retain
 *          their original insertion order.
 */

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

/**
 * @class SelfOrganizingList
 * @brief Singly linked list that keeps higher-priority items near the front.
 *
 * @tparam T Element type. Explicit instantiations are provided for int and
 *           std::string in SelfOrganizingList.cpp.
 *
 * Complexity:
 *   Insert / UpdatePriority / Find : O(n)
 *   Pop / Empty                    : O(1)
 *   Size / ToVector                : O(n)
 *   Space                          : O(n)
 */
template <typename T>
class SelfOrganizingList {
private:
    struct Node {
        T Data;
        int Priority;
        std::size_t Sequence;
        Node* Next;

        Node(const T& Item, int PriorityValue, std::size_t SequenceValue);
    };

    Node* Head;
    std::size_t NextSequence;

    void Clear() noexcept;
    void InsertNode(Node* NewNode);
    static bool ShouldComeBefore(const Node* Candidate, const Node* Current);

public:
    SelfOrganizingList();
    ~SelfOrganizingList();

    SelfOrganizingList(const SelfOrganizingList&) = delete;
    SelfOrganizingList& operator=(const SelfOrganizingList&) = delete;

    SelfOrganizingList(SelfOrganizingList&& Other) noexcept;
    SelfOrganizingList& operator=(SelfOrganizingList&& Other) noexcept;

    void Insert(const T& Item, int Priority);
    T Pop();
    bool UpdatePriority(const T& Item, int Priority);
    [[nodiscard]] bool Find(const T& Item) const;
    [[nodiscard]] std::vector<T> ToVector() const;
    [[nodiscard]] std::vector<std::pair<T, int>> ToPriorityVector() const;
    [[nodiscard]] bool Empty() const noexcept;
    [[nodiscard]] std::size_t Size() const noexcept;
};
