/**
 * @file test_PriorityProducerConsumerQueue.cpp
 * @brief Google Test suite for bucketed priority PriorityProducerConsumerQueue<T>.
 */

#include "PriorityProducerConsumerQueue.hpp"
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Basic operations
// ---------------------------------------------------------------------------

TEST(PriorityProducerConsumerQueueTest, ProduceAndConsumeSingleInt) {
    PriorityProducerConsumerQueue<int> q(5);

    ASSERT_TRUE(q.Produce(101, 2, 42));

    auto val = q.Consume();
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(val->id, 101);
    EXPECT_EQ(val->priority, 2);
    EXPECT_EQ(val->data, 42);
}

TEST(PriorityProducerConsumerQueueTest, RejectsInvalidPriority) {
    PriorityProducerConsumerQueue<int> q(5, 4);

    EXPECT_FALSE(q.Produce(1, -1, 10));
    EXPECT_FALSE(q.Produce(2, 4, 20));
    EXPECT_EQ(q.Size(), 0u);
}

TEST(PriorityProducerConsumerQueueTest, RejectsDuplicateIdWithoutConsumingCapacity) {
    PriorityProducerConsumerQueue<int> q(1);

    EXPECT_TRUE(q.Produce(1, 0, 10));
    EXPECT_FALSE(q.Produce(1, 3, 99));

    std::atomic<bool> second_item_produced{false};
    std::jthread producer([&](std::stop_token) {
        q.Produce(2, 1, 20);
        second_item_produced.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(second_item_produced.load(std::memory_order_acquire));

    auto first = q.Consume();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->id, 1);

    producer.join();
    EXPECT_TRUE(second_item_produced.load(std::memory_order_acquire));

    auto second = q.Consume();
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(second->id, 2);
}

// ---------------------------------------------------------------------------
// Priority ordering
// ---------------------------------------------------------------------------

TEST(PriorityProducerConsumerQueueTest, ConsumesHighestPriorityFirst) {
    PriorityProducerConsumerQueue<std::string> q(10, 4);

    ASSERT_TRUE(q.Produce(1, 0, "low"));
    ASSERT_TRUE(q.Produce(2, 3, "critical"));
    ASSERT_TRUE(q.Produce(3, 1, "normal"));

    auto first = q.Consume();
    auto second = q.Consume();
    auto third = q.Consume();

    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    ASSERT_TRUE(third.has_value());

    EXPECT_EQ(first->data, "critical");
    EXPECT_EQ(second->data, "normal");
    EXPECT_EQ(third->data, "low");
}

TEST(PriorityProducerConsumerQueueTest, SamePriorityPreservesFifoOrder) {
    PriorityProducerConsumerQueue<int> q(10, 4);

    ASSERT_TRUE(q.Produce(10, 2, 100));
    ASSERT_TRUE(q.Produce(11, 2, 200));
    ASSERT_TRUE(q.Produce(12, 2, 300));

    EXPECT_EQ(q.Consume()->data, 100);
    EXPECT_EQ(q.Consume()->data, 200);
    EXPECT_EQ(q.Consume()->data, 300);
}

TEST(PriorityProducerConsumerQueueTest, UpdatePriorityPromotesExistingItem) {
    PriorityProducerConsumerQueue<std::string> q(10, 4);

    ASSERT_TRUE(q.Produce(1, 0, "ordinary"));
    ASSERT_TRUE(q.Produce(2, 2, "soon"));

    EXPECT_TRUE(q.UpdatePriority(1, 3));

    auto first = q.Consume();
    auto second = q.Consume();

    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());

    EXPECT_EQ(first->id, 1);
    EXPECT_EQ(first->priority, 3);
    EXPECT_EQ(first->data, "ordinary");
    EXPECT_EQ(second->id, 2);
}

TEST(PriorityProducerConsumerQueueTest, UpdatePriorityReturnsFalseForMissingOrInvalidItems) {
    PriorityProducerConsumerQueue<int> q(10, 4);

    EXPECT_FALSE(q.UpdatePriority(99, 3));

    ASSERT_TRUE(q.Produce(1, 1, 10));
    EXPECT_FALSE(q.UpdatePriority(1, -1));
    EXPECT_FALSE(q.UpdatePriority(1, 4));
    EXPECT_TRUE(q.UpdatePriority(1, 3));
}

// ---------------------------------------------------------------------------
// Size / capacity accessors
// ---------------------------------------------------------------------------

TEST(PriorityProducerConsumerQueueTest, SizeAndContainsTrackLiveItems) {
    PriorityProducerConsumerQueue<int> q(10);

    EXPECT_EQ(q.Size(), 0u);
    EXPECT_FALSE(q.Contains(1));

    ASSERT_TRUE(q.Produce(1, 0, 10));
    ASSERT_TRUE(q.Produce(2, 1, 20));

    EXPECT_EQ(q.Size(), 2u);
    EXPECT_TRUE(q.Contains(1));
    EXPECT_TRUE(q.Contains(2));

    auto val = q.Consume();
    ASSERT_TRUE(val.has_value());

    EXPECT_EQ(q.Size(), 1u);
    EXPECT_FALSE(q.Contains(val->id));
}

TEST(PriorityProducerConsumerQueueTest, MaxSizeAndPriorityLevelsAccessors) {
    PriorityProducerConsumerQueue<int> q(7, 8);

    EXPECT_EQ(q.MaxSize(), 7u);
    EXPECT_EQ(q.PriorityLevels(), 8u);
}

// ---------------------------------------------------------------------------
// Bounded capacity enforcement
// ---------------------------------------------------------------------------

TEST(PriorityProducerConsumerQueueTest, BoundedCapacityBlocks) {
    constexpr std::size_t kCapacity = 3;
    PriorityProducerConsumerQueue<int> q(kCapacity);

    for (std::size_t i = 0; i < kCapacity; ++i) {
        ASSERT_TRUE(q.Produce(static_cast<int>(i), 0, static_cast<int>(i)));
    }
    EXPECT_EQ(q.Size(), kCapacity);

    std::atomic<bool> extra_produced{false};
    std::jthread blocker([&](std::stop_token) {
        q.Produce(99, 3, 99);
        extra_produced.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(extra_produced.load(std::memory_order_acquire));

    auto val = q.Consume();
    ASSERT_TRUE(val.has_value());

    blocker.join();
    EXPECT_TRUE(extra_produced.load(std::memory_order_acquire));
    EXPECT_EQ(q.Size(), kCapacity);
}

// ---------------------------------------------------------------------------
// Concurrent producer-consumer synchronization
// ---------------------------------------------------------------------------

TEST(PriorityProducerConsumerQueueTest, ConcurrentProducerConsumerSum) {
    constexpr int kN = 200;
    PriorityProducerConsumerQueue<int> q(8);
    std::atomic<int> sum{0};

    std::jthread producer([&](std::stop_token) {
        for (int i = 1; i <= kN; ++i) {
            q.Produce(i, i % 4, i);
        }
    });

    std::jthread consumer([&](std::stop_token) {
        for (int i = 0; i < kN; ++i) {
            auto v = q.Consume();
            if (v) sum += v->data;
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(sum.load(), kN * (kN + 1) / 2);
}

TEST(PriorityProducerConsumerQueueTest, MultipleProducersMultipleConsumers) {
    constexpr int kItemsPerProducer = 50;
    constexpr int kNumProducers = 4;
    constexpr int kTotalItems = kNumProducers * kItemsPerProducer;

    PriorityProducerConsumerQueue<int> q(10);
    std::atomic<int> consumed{0};

    auto producer_fn = [&](std::stop_token, int producer_id) {
        for (int i = 0; i < kItemsPerProducer; ++i) {
            const int id = producer_id * 1000 + i;
            q.Produce(id, i % 4, id);
        }
    };

    auto consumer_fn = [&](std::stop_token, int count) {
        for (int i = 0; i < count; ++i) {
            (void)q.Consume();
            consumed++;
        }
    };

    std::vector<std::jthread> threads;
    for (int p = 0; p < kNumProducers; ++p) {
        threads.emplace_back(producer_fn, p);
    }

    int per_consumer = kTotalItems / 2;
    threads.emplace_back(consumer_fn, per_consumer);
    threads.emplace_back(consumer_fn, kTotalItems - per_consumer);

    threads.clear();

    EXPECT_EQ(consumed.load(), kTotalItems);
}

// ---------------------------------------------------------------------------
// Template with move-only type
// ---------------------------------------------------------------------------

TEST(PriorityProducerConsumerQueueTest, MoveOnlyType) {
    PriorityProducerConsumerQueue<std::unique_ptr<int>> q(4);

    ASSERT_TRUE(q.Produce(1, 3, std::make_unique<int>(7)));

    auto val = q.Consume();
    ASSERT_TRUE(val.has_value());
    ASSERT_NE(val->data, nullptr);
    EXPECT_EQ(*val->data, 7);
}
