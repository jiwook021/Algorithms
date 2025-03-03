/**
 * @file test_UnsafeQueue.cpp
 * @brief Unit tests for the UnsafeQueue class (single-threaded only).
 *
 * Tests verify single-threaded correctness. Multi-threaded tests are
 * intentionally omitted since the queue has deliberate data races.
 */

#include "UnsafeQueue.hpp"
#include <cassert>
#include <iostream>

/** @brief Verify single-threaded produce and consume. */
static void TestProduceAndConsumeSingleThread() {
    UnsafeQueue q;
    q.Produce(10);
    q.Produce(20);

    int out = 0;
    assert(q.Consume(out));
    assert(out == 10);

    assert(q.Consume(out));
    assert(out == 20);
    std::cout << "[PASS] TestProduceAndConsumeSingleThread\n";
}

/** @brief Verify queue is empty after consuming all items. */
static void TestEmptyAfterConsumingAll() {
    UnsafeQueue q;
    q.Produce(1);

    int out = 0;
    q.Consume(out);
    assert(q.Empty());
    std::cout << "[PASS] TestEmptyAfterConsumingAll\n";
}

/** @brief Verify FIFO ordering with multiple items. */
static void TestProduceMultipleItems() {
    UnsafeQueue q;
    for (int i = 0; i < 10; ++i) q.Produce(i);

    for (int i = 0; i < 10; ++i) {
        int out = -1;
        assert(q.Consume(out));
        assert(out == i);
    }
    std::cout << "[PASS] TestProduceMultipleItems\n";
}

/** @brief Verify Empty() returns true for new queue. */
static void TestInitiallyEmpty() {
    UnsafeQueue q;
    assert(q.Empty());
    std::cout << "[PASS] TestInitiallyEmpty\n";
}

int main() {
    TestProduceAndConsumeSingleThread();
    TestEmptyAfterConsumingAll();
    TestProduceMultipleItems();
    TestInitiallyEmpty();

    std::cout << "\nAll UnsafeQueue tests passed.\n";
    return 0;
}
