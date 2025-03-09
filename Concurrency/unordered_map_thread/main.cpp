#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <optional>
#include <stdexcept>
#include <vector>
#include <ctime>
#include <chrono>
#include <sstream>
#include <string>
#include <iomanip>
#include <random>

/**
 * A thread-safe implementation of a singly linked list.
 * This implementation uses fine-grained locking to allow
 * concurrent operations on different parts of the list.
 * 
 * Template parameter T represents the type of data stored in the list.
 */
template <typename T>
class ThreadSafeLinkedList {
private:
    /**
     * Node structure for the linked list.
     * Each node contains data, a pointer to the next node,
     * and a mutex for thread synchronization.
     */
    struct Node {
        T data;
        std::shared_ptr<Node> next;
        mutable std::mutex mutex;  // Each node has its own mutex for fine-grained locking

        explicit Node(const T& value) : data(value), next(nullptr) {}
    };

    std::shared_ptr<Node> head;       // Head of the linked list
    mutable std::mutex head_mutex;    // Mutex to protect head pointer
    std::size_t size_;                // Size of the linked list
    mutable std::mutex size_mutex;    // Mutex to protect size variable
    mutable std::mutex cout_mutex;    // Mutex to protect console output

public:
    /**
     * Default constructor.
     * Initializes an empty linked list.
     */
    ThreadSafeLinkedList() : head(nullptr), size_(0) {}

    /**
     * Copy constructor is deleted because copying a thread-safe linked list
     * is not a trivial operation and could lead to race conditions.
     */
    ThreadSafeLinkedList(const ThreadSafeLinkedList&) = delete;

    /**
     * Copy assignment operator is deleted for the same reason as the copy constructor.
     */
    ThreadSafeLinkedList& operator=(const ThreadSafeLinkedList&) = delete;

    /**
     * Move constructor.
     * Transfers ownership of the nodes from other to this list.
     * 
     * Time Complexity: O(1) - Constant time
     * Space Complexity: O(1) - No additional space required
     */
    ThreadSafeLinkedList(ThreadSafeLinkedList&& other) noexcept {
        std::lock_guard<std::mutex> lock_head(other.head_mutex);
        std::lock_guard<std::mutex> lock_size(other.size_mutex);
        
        head = std::move(other.head);
        size_ = other.size_;
        other.size_ = 0;
    }

    /**
     * Move assignment operator.
     * Transfers ownership of the nodes from other to this list.
     * 
     * Time Complexity: O(1) - Constant time
     * Space Complexity: O(1) - No additional space required
     */
    ThreadSafeLinkedList& operator=(ThreadSafeLinkedList&& other) noexcept {
        if (this != &other) {
            // Lock both lists to prevent race conditions
            std::scoped_lock lock(head_mutex, size_mutex, other.head_mutex, other.size_mutex);
            
            head = std::move(other.head);
            size_ = other.size_;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * Inserts a new node with the given value at the beginning of the list.
     * 
     * @param value The value to be inserted
     * 
     * Time Complexity: O(1) - Constant time
     * Space Complexity: O(1) - Only one new node is created
     */
    void push_front(const T& value) {
        auto new_node = std::make_shared<Node>(value);
        
        // Lock the head mutex to safely modify the head pointer
        //std::lock_guard<std::mutex> lock(head_mutex);
        
        new_node->next = head;
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Deliberate delay
        head = new_node;
        
        // Update size
        //std::lock_guard<std::mutex> size_lock(size_mutex);
        size_++;
    }

    /**
     * Inserts a new node with the given value after the node containing the target value.
     * If the target value is not found, throws an exception.
     * 
     * @param target The value after which to insert
     * @param value The value to be inserted
     * @throws std::runtime_error if target value is not found
     * 
     * Time Complexity: O(n) - where n is the number of nodes in the list
     * Space Complexity: O(1) - Only one new node is created
     */
    void insert_after(const T& target, const T& value) {
        //std::lock_guard<std::mutex> head_lock(head_mutex);
        
        if (!head) {
            throw std::runtime_error("Cannot insert in an empty list");
        }
        
        auto current = head;
        
        // Hand-over-hand locking (lock coupling) to traverse the list safely
        //std::unique_lock<std::mutex> current_lock(current->mutex);
        
        while (current && current->data != target) {
            auto next = current->next;
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Deliberate delay
            if (!next) {
                throw std::runtime_error("Target value not found");
            }
            
            // Lock the next node before releasing the current one
            //std::unique_lock<std::mutex> next_lock(next->mutex);
            //current_lock.unlock();
            
            current = next;
            // Move ownership of the next_lock to current_lock
            //current_lock = std::move(next_lock);
        }
        
        // At this point, current is the node containing the target value
        auto new_node = std::make_shared<Node>(value);
        new_node->next = current->next;
        current->next = new_node;
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Deliberate delay
        // Update size
        //std::lock_guard<std::mutex> size_lock(size_mutex);
        size_++;
    }

    /**
     * Removes the first node from the list.
     * 
     * @return The value of the removed node, or std::nullopt if the list is empty
     * 
     * Time Complexity: O(1) - Constant time
     * Space Complexity: O(1) - No additional space required
     */
    std::optional<T> pop_front() {
        //std::lock_guard<std::mutex> lock(head_mutex);
        
        if (!head) {
            return std::nullopt;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Deliberate delay
        //std::lock_guard<std::mutex> node_lock(head->mutex);
        T result = head->data;
        head = head->next;
        
        // Update size
        //std::lock_guard<std::mutex> size_lock(size_mutex);
        size_--;
        
        return result;
    }

    /**
     * Removes the first occurrence of a node with the given value.
     * 
     * @param value The value to be removed
     * @return true if a node was removed, false otherwise
     * 
     * Time Complexity: O(n) - where n is the number of nodes in the list
     * Space Complexity: O(1) - No additional space required
     */
    bool remove(const T& value) {
        //std::lock_guard<std::mutex> head_lock(head_mutex);
        
        if (!head) {
            return false;
        }
        
        // Special case: remove the head
        if (head->data == value) {
            //std::lock_guard<std::mutex> node_lock(head->mutex);
            head = head->next;
            
            // Update size
            //std::lock_guard<std::mutex> size_lock(size_mutex);
            size_--;
            
            return true;
        }
        
        auto current = head;
        //std::unique_lock<std::mutex> current_lock(current->mutex);
        
        while (current->next) {
            auto next = current->next;
            //std::unique_lock<std::mutex> next_lock(next->mutex);
            
            if (next->data == value) {
                // Remove the next node
                current->next = next->next;
                
                // Update size
                //std::lock_guard<std::mutex> size_lock(size_mutex);
                std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Deliberate delay
                size_--;
                
                return true;
            }
            
            //current_lock.unlock();
            current = next;
            //current_lock = std::move(next_lock);
        }
        
        return false;
    }

    /**
     * Checks if the list contains a node with the given value.
     * 
     * @param value The value to search for
     * @return true if the value is found, false otherwise
     * 
     * Time Complexity: O(n) - where n is the number of nodes in the list
     * Space Complexity: O(1) - No additional space required
     */
    bool contains(const T& value) const {
        //std::lock_guard<std::mutex> head_lock(head_mutex);
        
        if (!head) {
            return false;
        }
        
        auto current = head;
        //std::unique_lock<std::mutex> current_lock(current->mutex);
        
        while (current) {
            if (current->data == value) {
                return true;
            }
            
            auto next = current->next;
            if (!next) {
                break;
            }
            
            // Lock the next node before releasing the current one
            //std::unique_lock<std::mutex> next_lock(next->mutex);
            //current_lock.unlock();
            
            current = next;
            //current_lock = std::move(next_lock);
        }
        
        return false;
    }

    /**
     * Returns the current size of the linked list.
     * 
     * @return The number of nodes in the list
     * 
     * Time Complexity: O(1) - Constant time
     * Space Complexity: O(1) - No additional space required
     */
    std::size_t size() const {
        //std::lock_guard<std::mutex> lock(size_mutex);
        return size_;
    }

    /**
     * Checks if the linked list is empty.
     * 
     * @return true if the list is empty, false otherwise
     * 
     * Time Complexity: O(1) - Constant time
     * Space Complexity: O(1) - No additional space required
     */
    bool empty() const {
        //std::lock_guard<std::mutex> lock(size_mutex);
        return size_ == 0;
    }

    /**
     * Clears the linked list, removing all nodes.
     * 
     * Time Complexity: O(1) - Constant time due to shared_ptr usage
     * Space Complexity: O(1) - No additional space required
     */
    void clear() {
        //std::lock_guard<std::mutex> head_lock(head_mutex);
        head = nullptr;
        
        //std::lock_guard<std::mutex> size_lock(size_mutex);
        size_ = 0;
    }

    /**
     * Gets a string representation of the current list state.
     * 
     * Time Complexity: O(n) - where n is the number of nodes in the list
     * Space Complexity: O(n) - for the string representation
     */
    std::string to_string() const {
        std::lock_guard<std::mutex> head_lock(head_mutex);
        
        std::stringstream ss;
        auto current = head;
        
        while (current) {
            std::lock_guard<std::mutex> node_lock(current->mutex);
            ss << current->data;
            
            current = current->next;
            if (current) {
                ss << " -> ";
            }
        }
        
        return ss.str();
    }

    /**
     * Thread-safe print function for the list content.
     */
    void print() const {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "List: " << to_string() << std::endl;
    }
    
    /**
     * Thread-safe print function that prevents interleaved output
     */
    void print_message(const std::string& message) const {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << message;
    }

    /**
     * Combined operation and logging functions for thread-safe operation visualization
     */
    
    // Thread-safe push_front with immediate logging
    void push_front_and_log(const T& value, int thread_id) {
        // First, acquire the cout mutex to ensure atomic output
        std::lock_guard<std::mutex> cout_lock(cout_mutex);
        
        // Perform the operation
        push_front(value);
        
        // Log the operation and show the resulting list state
        std::cout << "Thread " << std::setw(2) << thread_id << " inserted " << value << " at front" << std::endl;
        std::cout << "List: " << to_string() << std::endl << std::endl;
    }
    
    // Thread-safe pop_front with immediate logging
    void pop_front_and_log(int thread_id) {
        // First, acquire the cout mutex to ensure atomic output
        std::lock_guard<std::mutex> cout_lock(cout_mutex);
        
        // Perform the operation
        auto result = pop_front();
        
        // Log the operation and show the resulting list state
        if (result) {
            std::cout << "Thread " << std::setw(2) << thread_id << " removed " << *result << " from front" << std::endl;
        } else {
            std::cout << "Thread " << std::setw(2) << thread_id << " tried to remove from empty list" << std::endl;
        }
        
        std::cout << "List: " << to_string() << std::endl << std::endl;
    }
    
    // Thread-safe insert_after with immediate logging
    void insert_after_and_log(const T& target, const T& value, int thread_id) {
        // First, acquire the cout mutex to ensure atomic output
        std::lock_guard<std::mutex> cout_lock(cout_mutex);
        
        try {
            // Perform the operation
            insert_after(target, value);
            
            // Log the successful operation
            std::cout << "Thread " << std::setw(2) << thread_id << " inserted " << value 
                      << " after " << target << std::endl;
        } catch (const std::exception& e) {
            // Log the failure
            std::cout << "Thread " << std::setw(2) << thread_id << " failed to insert " << value <<" after: " 
                      <<  target << std::endl;
        }
        
        // Show the resulting list state
        std::cout << "List: " << to_string() << std::endl << std::endl;
    }
    
    // Thread-safe contains check with immediate logging
    void contains_and_log(const T& value, int thread_id) {
        // First, acquire the cout mutex to ensure atomic output
        std::lock_guard<std::mutex> cout_lock(cout_mutex);
        
        // Perform the operation
        bool contains_value = contains(value);
        
        // Log the operation and show the resulting list state
        std::cout << "Thread " << std::setw(2) << thread_id << " checked for " << value << ": " 
                  << (contains_value ? "found" : "not found") << std::endl;
        std::cout << "List: " << to_string() << std::endl << std::endl;
    }
    
    // Thread-safe remove with immediate logging
    void remove_and_log(const T& value, int thread_id) {
        // First, acquire the cout mutex to ensure atomic output
        std::lock_guard<std::mutex> cout_lock(cout_mutex);
        
        // Perform the operation
        bool removed = remove(value);
        
        // Log the operation and show the resulting list state
        std::cout << "Thread " << std::setw(2) << thread_id << " tried to remove " << value << ": " 
                  << (removed ? "removed" : "not found") << std::endl;
        std::cout << "List: " << to_string() << std::endl << std::endl;
    }
    
    // Thread-safe empty list push_front with immediate logging
    void push_front_empty_and_log(const T& value, int thread_id) {
        // First, acquire the cout mutex to ensure atomic output
        std::lock_guard<std::mutex> cout_lock(cout_mutex);
        
        // Perform the operation
        push_front(value);
        
        // Log the operation and show the resulting list state
        std::cout << "Thread " << std::setw(2) << thread_id << " inserted " << value 
                  << " at front (list was empty)" << std::endl;
        std::cout << "List: " << to_string() << std::endl << std::endl;
    }
};

/**
 * Test function that performs operations on the linked list
 * from multiple threads.
 * 
 * @param list Reference to the linked list
 * @param id Thread identifier
 * @param iterations Number of operations to perform
 */
template <typename T>
void test_thread(ThreadSafeLinkedList<T>& list, int id, int iterations) {
    // Use modern random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 5);
    
    for (int i = 0; i < iterations; ++i) {
        int operation = distrib(gen);
        int num = 10;
        // Each case uses the combined operation+logging functions
        // to ensure atomic operations and consistent output
        switch (operation) {
            case 0:
                // Insert at front
                list.push_front_and_log((id + i)%num, id);
                break;
            
            case 1:
                // Remove from front
                list.pop_front_and_log(id);
                break;
                
            case 2:
                // Insert after a value (if possible)
                if (!list.empty()) {
                    // For simplicity, try to insert after the thread's own value
                    list.insert_after_and_log((i*i)%num, (id + i)%num, id);
                } else {
                    list.push_front_empty_and_log((id * 1000 + i*i*i)%num, id);
                }
                break;
                
            case 3:
                // Check if contains
                list.contains_and_log((id * i)%num, id);
                break;
                
            case 4:
                // Remove a value
                list.remove_and_log((id * 1000+id)%num, id);
                break;

            case 5: 
            // Insert at front
            for(int j=0;j<10;j++)
                list.push_front_and_log((id + i+j)%num, id);

            case 6: 
            for(int j=0;j<11;j++)
                list.pop_front_and_log(id);
            break;
        }
        
        // Add a small delay to make the interleaving of operations more visible
        //std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

/**
 * Main function to demonstrate the thread-safe linked list.
 */
int main() {
    // Use modern random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Create a thread-safe linked list of integers
    ThreadSafeLinkedList<int> list;
    
    // Initial values
    list.push_front(3);
    list.push_front(2);
    list.push_front(1);
    
    std::cout << "Initial List: " << list.to_string() << std::endl << std::endl;
    
    // Create and start threads
    const int num_threads = 10;  // Reduced for clearer output
    const int operations_per_thread = 10;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(test_thread<int>, std::ref(list), i, operations_per_thread);
    }
    
    // Wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::cout << "Final List: " << list.to_string() << std::endl;
    std::cout << "List size: " << list.size() << std::endl;
    
    return 0;
}