/**
 * @file ThreadSafeQueue.h
 * @brief Thread-safe queue implementation for Producer-Consumer pattern
 */

 #pragma once

 #include <queue>
 #include <mutex>
 #include <condition_variable>
 #include <optional>
 #include <chrono>
 #include <atomic>
 
 /**
  * @class ThreadSafeQueue
  * @brief Template class implementing a thread-safe queue for Producer-Consumer pattern
  * 
  * This class provides thread-safe operations to push and pop elements from a queue.
  * It uses mutex for thread synchronization to ensure that multiple threads can safely
  * access the queue without race conditions.
  * 
  * @tparam T Type of elements stored in the queue
  */
 template <typename T>
 class ThreadSafeQueue {
 public:
     /**
      * @brief Default constructor
      */
     ThreadSafeQueue() = default;
 
     /**
      * @brief Deleted copy constructor to prevent accidental copying
      */
     ThreadSafeQueue(const ThreadSafeQueue&) = delete;
 
     /**
      * @brief Deleted copy assignment to prevent accidental copying
      */
     ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;
 
     /**
      * @brief Push an element to the queue
      * 
      * This method acquires a lock on the mutex to ensure thread-safety,
      * pushes the element to the queue, and notifies one waiting thread.
      * 
      * @param value The element to push
      */
     void push(T value) {
         {
             // Lock is used here to protect the shared queue from concurrent access
             // This prevents data races when multiple threads try to push simultaneously
             std::lock_guard<std::mutex> lock(m_mutex);
             m_queue.push(std::move(value));
         }
         // Notify one waiting thread that data is available
         m_cv.notify_one();
     }
 
     /**
      * @brief Pop an element from the queue (blocking)
      * 
      * This method blocks until an element is available, then returns it.
      * 
      * @return The popped element
      */
     T pop() {
         // Unique_lock is used instead of lock_guard because we need to unlock temporarily
         // during condition variable waiting
         std::unique_lock<std::mutex> lock(m_mutex);
         
         // Wait until queue is not empty
         m_cv.wait(lock, [this] { return !m_queue.empty() || m_done; });
         
         // If queue is empty and done flag is set, throw exception
         if (m_queue.empty() && m_done) {
             throw std::runtime_error("Queue is closed");
         }
         
         // Get the front value
         T value = std::move(m_queue.front());
         m_queue.pop();
         
         return value;
     }
 
     /**
      * @brief Try to pop an element from the queue (non-blocking)
      * 
      * This method does not block, but returns an empty optional if no element is available.
      * 
      * @return An optional containing the popped element if available, empty otherwise
      */
     std::optional<T> try_pop() {
         std::lock_guard<std::mutex> lock(m_mutex);
         
         if (m_queue.empty()) {
             return std::nullopt;
         }
         
         T value = std::move(m_queue.front());
         m_queue.pop();
         
         return value;
     }
 
     /**
      * @brief Try to pop an element from the queue with timeout
      * 
      * This method blocks until an element is available or timeout occurs.
      * 
      * @param timeout Maximum time to wait in milliseconds
      * @return An optional containing the popped element if available, empty otherwise
      */
     std::optional<T> try_pop_for(std::chrono::milliseconds timeout) {
         std::unique_lock<std::mutex> lock(m_mutex);
         
         // Wait until queue is not empty or timeout
         m_cv.wait_for(lock, timeout, [this] { return !m_queue.empty() || m_done; });
         
         // If queue is empty, return empty optional
         if (m_queue.empty()) {
             return std::nullopt;
         }
         
         T value = std::move(m_queue.front());
         m_queue.pop();
         
         return value;
     }
 
     /**
      * @brief Check if the queue is empty
      * 
      * @return true if the queue is empty, false otherwise
      */
     bool empty() const {
         std::lock_guard<std::mutex> lock(m_mutex);
         return m_queue.empty();
     }
 
     /**
      * @brief Get the size of the queue
      * 
      * @return The number of elements in the queue
      */
     size_t size() const {
         std::lock_guard<std::mutex> lock(m_mutex);
         return m_queue.size();
     }
 
     /**
      * @brief Mark the queue as done (no more elements will be pushed)
      * 
      * This method is used to signal waiting threads that no more elements will be pushed.
      */
     void done() {
         {
             std::lock_guard<std::mutex> lock(m_mutex);
             m_done = true;
         }
         m_cv.notify_all();
     }
 
     /**
      * @brief Check if the queue is done
      * 
      * @return true if the queue is done, false otherwise
      */
     bool is_done() const {
         std::lock_guard<std::mutex> lock(m_mutex);
         return m_done && m_queue.empty();
     }
 
 private:
     std::queue<T> m_queue;                  ///< Underlying queue
     mutable std::mutex m_mutex;             ///< Mutex for thread synchronization
     std::condition_variable m_cv;           ///< Condition variable for waiting
     std::atomic<bool> m_done{false};        ///< Flag indicating if the queue is done
 };