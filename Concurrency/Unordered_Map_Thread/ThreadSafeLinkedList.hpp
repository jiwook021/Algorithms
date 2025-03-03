/**
 * @file ThreadSafeLinkedList.hpp
 * @brief Singly linked list with per-node shared mutexes for read/write separation.
 *
 * @details
 * This class implements a thread-safe singly linked list using std::shared_mutex
 * to allow concurrent readers while serializing writers. Write operations
 * (PushFront, PopFront, Remove, InsertAfter, Clear) acquire std::unique_lock.
 * Read operations (Contains, Size, Empty, ToString) acquire std::shared_lock.
 *
 * Complexity:
 *   - PushFront():    O(1)
 *   - PopFront():     O(1)
 *   - Contains():     O(n)
 *   - Remove():       O(n)
 *   - InsertAfter():  O(n)
 */

#ifndef THREAD_SAFE_LINKED_LIST_HPP
#define THREAD_SAFE_LINKED_LIST_HPP

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <chrono>

/**
 * @class ThreadSafeLinkedList
 * @brief Singly linked list with std::shared_mutex for read/write separation.
 * @tparam T Element type.
 */
template <typename T>
class ThreadSafeLinkedList {
private:
    /** @brief Linked list node. */
    struct Node {
        T data;                          ///< Element data.
        std::shared_ptr<Node> next;      ///< Next node.
        mutable std::shared_mutex mutex; ///< Per-node read/write lock.
        /** @brief Construct a node with given value. */
        explicit Node(const T& value) : data(value), next(nullptr) {}
    };

    std::shared_ptr<Node> head_;              ///< Head pointer.
    mutable std::shared_mutex head_mutex_;     ///< Head guard (read/write).
    std::size_t size_;                         ///< Element count.
    mutable std::shared_mutex size_mutex_;     ///< Size guard (read/write).
    mutable std::mutex cout_mutex_;            ///< Output guard.

public:
    /** @brief Construct an empty list. */
    ThreadSafeLinkedList() : head_(nullptr), size_(0) {}

    /** @brief Non-copyable. */
    ThreadSafeLinkedList(const ThreadSafeLinkedList&) = delete;
    /** @brief Non-copyable. */
    ThreadSafeLinkedList& operator=(const ThreadSafeLinkedList&) = delete;

    /**
     * @brief Move constructor.
     * @param other List to move from.
     */
    ThreadSafeLinkedList(ThreadSafeLinkedList&& other) noexcept {
        std::unique_lock<std::shared_mutex> lock_head(other.head_mutex_);
        std::unique_lock<std::shared_mutex> lock_size(other.size_mutex_);
        head_ = std::move(other.head_);
        size_ = other.size_;
        other.size_ = 0;
    }

    /**
     * @brief Move assignment operator.
     * @param other List to move from.
     * @return Reference to this.
     */
    ThreadSafeLinkedList& operator=(ThreadSafeLinkedList&& other) noexcept {
        if (this != &other) {
            std::scoped_lock lock(head_mutex_, size_mutex_,
                                  other.head_mutex_, other.size_mutex_);
            head_ = std::move(other.head_);
            size_ = other.size_;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * @brief Insert a value at the front. O(1).
     * @param value Value to insert.
     */
    void PushFront(const T& value) {
        auto new_node = std::make_shared<Node>(value);
        std::unique_lock<std::shared_mutex> lock(head_mutex_);
        new_node->next = head_;
        head_ = new_node;
        std::unique_lock<std::shared_mutex> size_lock(size_mutex_);
        size_++;
    }

    /**
     * @brief Insert a value after the first occurrence of target. O(n).
     * @param target Value to search for.
     * @param value Value to insert.
     * @throws std::runtime_error if list is empty or target not found.
     */
    void InsertAfter(const T& target, const T& value) {
        std::unique_lock<std::shared_mutex> head_lock(head_mutex_);
        if (!head_) throw std::runtime_error("Cannot insert in an empty list");

        auto current = head_;
        while (current && current->data != target) {
            auto next = current->next;
            if (!next) throw std::runtime_error("Target value not found");
            current = next;
        }

        auto new_node = std::make_shared<Node>(value);
        new_node->next = current->next;
        current->next = new_node;
        std::unique_lock<std::shared_mutex> size_lock(size_mutex_);
        size_++;
    }

    /**
     * @brief Remove and return the front element. O(1).
     * @return The front value, or std::nullopt if empty.
     */
    std::optional<T> PopFront() {
        std::unique_lock<std::shared_mutex> lock(head_mutex_);
        if (!head_) return std::nullopt;
        T result = head_->data;
        head_ = head_->next;
        std::unique_lock<std::shared_mutex> size_lock(size_mutex_);
        size_--;
        return result;
    }

    /**
     * @brief Remove the first occurrence of a value. O(n).
     * @param value Value to remove.
     * @return True if the value was found and removed.
     */
    bool Remove(const T& value) {
        std::unique_lock<std::shared_mutex> head_lock(head_mutex_);
        if (!head_) return false;

        if (head_->data == value) {
            head_ = head_->next;
            std::unique_lock<std::shared_mutex> size_lock(size_mutex_);
            size_--;
            return true;
        }

        auto current = head_;
        while (current->next) {
            auto next = current->next;
            if (next->data == value) {
                current->next = next->next;
                std::unique_lock<std::shared_mutex> size_lock(size_mutex_);
                size_--;
                return true;
            }
            current = next;
        }
        return false;
    }

    /**
     * @brief Check if a value exists in the list. O(n).
     * @param value Value to search for.
     * @return True if found.
     */
    bool Contains(const T& value) const {
        std::shared_lock<std::shared_mutex> head_lock(head_mutex_);
        auto current = head_;
        while (current) {
            std::shared_lock<std::shared_mutex> node_lock(current->mutex);
            if (current->data == value) return true;
            current = current->next;
        }
        return false;
    }

    /**
     * @brief Return the number of elements.
     * @return Size.
     */
    std::size_t Size() const {
        std::shared_lock<std::shared_mutex> lock(size_mutex_);
        return size_;
    }

    /**
     * @brief Check if the list is empty.
     * @return True if empty.
     */
    bool Empty() const {
        std::shared_lock<std::shared_mutex> lock(size_mutex_);
        return size_ == 0;
    }

    /**
     * @brief Remove all elements.
     */
    void Clear() {
        std::unique_lock<std::shared_mutex> head_lock(head_mutex_);
        head_ = nullptr;
        std::unique_lock<std::shared_mutex> size_lock(size_mutex_);
        size_ = 0;
    }

    /**
     * @brief Convert the list to a string representation.
     * @return String like "1 -> 2 -> 3".
     */
    std::string ToString() const {
        std::shared_lock<std::shared_mutex> head_lock(head_mutex_);
        std::stringstream ss;
        auto current = head_;
        while (current) {
            std::shared_lock<std::shared_mutex> node_lock(current->mutex);
            ss << current->data;
            current = current->next;
            if (current) ss << " -> ";
        }
        return ss.str();
    }

    /**
     * @brief Print the list to stdout.
     */
    void Print() const {
        std::lock_guard<std::mutex> lock(cout_mutex_);
        std::cout << "List: " << ToString() << std::endl;
    }

    /**
     * @brief Print a message to stdout (thread-safe).
     * @param message Message to print.
     */
    void PrintMessage(const std::string& message) const {
        std::lock_guard<std::mutex> lock(cout_mutex_);
        std::cout << message;
    }

    /**
     * @brief Push a value and log the operation.
     * @param value Value to insert.
     * @param thread_id Thread identifier for logging.
     */
    void PushFrontAndLog(const T& value, int thread_id) {
        std::lock_guard<std::mutex> cout_lock(cout_mutex_);
        PushFront(value);
        std::cout << "Thread " << std::setw(2) << thread_id
                  << " inserted " << value << " at front\n";
        std::cout << "List: " << ToString() << "\n\n";
    }

    /**
     * @brief Pop front and log the operation.
     * @param thread_id Thread identifier for logging.
     */
    void PopFrontAndLog(int thread_id) {
        std::lock_guard<std::mutex> cout_lock(cout_mutex_);
        auto result = PopFront();
        if (result)
            std::cout << "Thread " << std::setw(2) << thread_id
                      << " removed " << *result << " from front\n";
        else
            std::cout << "Thread " << std::setw(2) << thread_id
                      << " tried to remove from empty list\n";
        std::cout << "List: " << ToString() << "\n\n";
    }

    /**
     * @brief Insert after target and log.
     * @param target Target value.
     * @param value Value to insert.
     * @param thread_id Thread identifier for logging.
     */
    void InsertAfterAndLog(const T& target, const T& value, int thread_id) {
        std::lock_guard<std::mutex> cout_lock(cout_mutex_);
        try {
            InsertAfter(target, value);
            std::cout << "Thread " << std::setw(2) << thread_id
                      << " inserted " << value << " after " << target << "\n";
        } catch (const std::exception&) {
            std::cout << "Thread " << std::setw(2) << thread_id
                      << " failed to insert " << value << " after " << target << "\n";
        }
        std::cout << "List: " << ToString() << "\n\n";
    }

    /**
     * @brief Check contains and log.
     * @param value Value to search for.
     * @param thread_id Thread identifier for logging.
     */
    void ContainsAndLog(const T& value, int thread_id) {
        std::lock_guard<std::mutex> cout_lock(cout_mutex_);
        bool found = Contains(value);
        std::cout << "Thread " << std::setw(2) << thread_id
                  << " checked for " << value << ": "
                  << (found ? "found" : "not found") << "\n";
        std::cout << "List: " << ToString() << "\n\n";
    }

    /**
     * @brief Remove value and log.
     * @param value Value to remove.
     * @param thread_id Thread identifier for logging.
     */
    void RemoveAndLog(const T& value, int thread_id) {
        std::lock_guard<std::mutex> cout_lock(cout_mutex_);
        bool removed = Remove(value);
        std::cout << "Thread " << std::setw(2) << thread_id
                  << " tried to remove " << value << ": "
                  << (removed ? "removed" : "not found") << "\n";
        std::cout << "List: " << ToString() << "\n\n";
    }

    /**
     * @brief Push front into empty list and log.
     * @param value Value to insert.
     * @param thread_id Thread identifier for logging.
     */
    void PushFrontEmptyAndLog(const T& value, int thread_id) {
        std::lock_guard<std::mutex> cout_lock(cout_mutex_);
        PushFront(value);
        std::cout << "Thread " << std::setw(2) << thread_id
                  << " inserted " << value << " at front (list was empty)\n";
        std::cout << "List: " << ToString() << "\n\n";
    }
};

#endif // THREAD_SAFE_LINKED_LIST_HPP
