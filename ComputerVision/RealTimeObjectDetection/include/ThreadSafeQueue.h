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
     void Push(T value) {
         {
             // Lock is used here to protect the shared queue from concurrent access
             // This prevents data races when multiple threads try to push simultaneously
             std::lock_guard<std::mutex> Lock(MMutex);
             MQueue.push(std::move(value));
         }
         // Notify one waiting thread that data is available
         MCv.notify_one();
     }
 
     /**
      * @brief Pop an element from the queue (blocking)
      * 
      * This method blocks until an element is available, then returns it.
      * 
      * @return The popped element
      */
     T Pop() {
         // Unique_lock is used instead of lock_guard because we need to unlock temporarily
         // during condition variable waiting
         std::unique_lock<std::mutex> Lock(MMutex);
         
         // Wait until queue is not empty
         MCv.wait(lock, [this] { return !MQueue.empty() || MDone; });
         
         // If queue is empty and done flag is set, throw exception
         if (MQueue.empty() && MDone) {
             throw std::runtime_error("Queue is closed");
         }
         
         // Get the front value
         T value = std::move(MQueue.front());
         MQueue.pop();
         
         return value;
     }
 
     /**
      * @brief Try to pop an element from the queue (non-blocking)
      * 
      * This method does not block, but returns an empty optional if no element is available.
      * 
      * @return An optional containing the popped element if available, empty otherwise
      */
     std::optional<T> TryPop() {
         std::lock_guard<std::mutex> Lock(MMutex);
         
         if (MQueue.empty()) {
             return std::nullopt;
         }
         
         T value = std::move(MQueue.front());
         MQueue.pop();
         
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
     std::optional<T> TryPopFor(std::chrono::milliseconds Timeout) {
         std::unique_lock<std::mutex> Lock(MMutex);
         
         // Wait until queue is not empty or timeout
         MCv.wait_for(lock, Timeout, [this] { return !MQueue.empty() || MDone; });
         
         // If queue is empty, return empty optional
         if (MQueue.empty()) {
             return std::nullopt;
         }
         
         T value = std::move(MQueue.front());
         MQueue.pop();
         
         return value;
     }
 
     /**
      * @brief Check if the queue is empty
      * 
      * @return true if the queue is empty, false otherwise
      */
     bool empty() const {
         std::lock_guard<std::mutex> Lock(MMutex);
         return MQueue.empty();
     }
 
     /**
      * @brief Get the size of the queue
      * 
      * @return The number of elements in the queue
      */
     size_t size() const {
         std::lock_guard<std::mutex> Lock(MMutex);
         return MQueue.size();
     }
 
     /**
      * @brief Mark the queue as done (no more elements will be pushed)
      * 
      * This method is used to signal waiting threads that no more elements will be pushed.
      */
     void Done() {
         {
             std::lock_guard<std::mutex> Lock(MMutex);
             MDone = true;
         }
         MCv.notify_all();
     }
 
     /**
      * @brief Check if the queue is done
      * 
      * @return true if the queue is done, false otherwise
      */
     bool IsDone() const {
         std::lock_guard<std::mutex> Lock(MMutex);
         return MDone && MQueue.empty();
     }
 
 private:
     std::queue<T> MQueue;                  ///< Underlying queue
     mutable std::mutex MMutex;             ///< Mutex for thread synchronization
     std::condition_variable MCv;           ///< Condition variable for waiting
     std::atomic<bool> MDone{false};        ///< Flag indicating if the queue is done
 };