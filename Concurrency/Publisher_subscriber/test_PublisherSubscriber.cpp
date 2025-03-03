/**
 * @file test_PublisherSubscriber.cpp
 * @brief Google Test unit tests for the Publisher-Subscriber pattern.
 *
 * Tests verify callback-based subscription, notification delivery,
 * unsubscription, fan-out to multiple subscribers, and concurrent
 * publish + subscribe safety.
 */

#include "PublisherSubscriber.hpp"
#include <gtest/gtest.h>

#include <atomic>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

/**
 * @brief Print a labelled section banner to stdout for readable test output.
 * @param title Short section name.
 * @param why   Rationale for the test.
 */
void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

// ---------------------------------------------------------------------------
// Basic publish-receive
// ---------------------------------------------------------------------------

TEST(PublisherSubscriberTest, BasicPublishReceive) {
    PrintSection("BasicPublishReceive",
                 "Verify a single subscriber receives the published message.");

    Publisher<std::string> pub;
    std::vector<std::string> received;

    pub.Subscribe([&received](const std::string& msg) {
        received.push_back(msg);
    });

    pub.Notify("hello");

    ASSERT_EQ(received.size(), 1u);
    EXPECT_EQ(received[0], "hello");
}

TEST(PublisherSubscriberTest, BasicPublishReceiveInt) {
    PrintSection("BasicPublishReceiveInt",
                 "Verify templatized Publisher works with non-string types.");

    Publisher<int> pub;
    std::vector<int> received;

    pub.Subscribe([&received](const int& val) {
        received.push_back(val);
    });

    pub.Notify(42);

    ASSERT_EQ(received.size(), 1u);
    EXPECT_EQ(received[0], 42);
}

// ---------------------------------------------------------------------------
// Multiple subscribers fan-out
// ---------------------------------------------------------------------------

TEST(PublisherSubscriberTest, MultipleSubscribersFanOut) {
    PrintSection("MultipleSubscribersFanOut",
                 "Verify all subscribers receive each published message.");

    Publisher<std::string> pub;
    std::vector<std::string> r1, r2, r3;

    pub.Subscribe([&r1](const std::string& msg) { r1.push_back(msg); });
    pub.Subscribe([&r2](const std::string& msg) { r2.push_back(msg); });
    pub.Subscribe([&r3](const std::string& msg) { r3.push_back(msg); });

    pub.Notify("event");

    ASSERT_EQ(r1.size(), 1u);
    ASSERT_EQ(r2.size(), 1u);
    ASSERT_EQ(r3.size(), 1u);
    EXPECT_EQ(r1[0], "event");
    EXPECT_EQ(r2[0], "event");
    EXPECT_EQ(r3[0], "event");
}

TEST(PublisherSubscriberTest, SubscriberCountTracking) {
    PrintSection("SubscriberCountTracking",
                 "Verify SubscriberCount reflects subscribe/unsubscribe operations.");

    Publisher<std::string> pub;
    EXPECT_EQ(pub.SubscriberCount(), 0u);

    auto id1 = pub.Subscribe([](const std::string&) {});
    EXPECT_EQ(pub.SubscriberCount(), 1u);

    auto id2 = pub.Subscribe([](const std::string&) {});
    EXPECT_EQ(pub.SubscriberCount(), 2u);

    pub.Unsubscribe(id1);
    EXPECT_EQ(pub.SubscriberCount(), 1u);

    pub.Unsubscribe(id2);
    EXPECT_EQ(pub.SubscriberCount(), 0u);
}

// ---------------------------------------------------------------------------
// Unsubscribe stops delivery
// ---------------------------------------------------------------------------

TEST(PublisherSubscriberTest, UnsubscribeStopsDelivery) {
    PrintSection("UnsubscribeStopsDelivery",
                 "Verify an unsubscribed callback no longer receives messages.");

    Publisher<std::string> pub;
    std::vector<std::string> r1, r2;

    auto id1 = pub.Subscribe([&r1](const std::string& msg) { r1.push_back(msg); });
    pub.Subscribe([&r2](const std::string& msg) { r2.push_back(msg); });

    pub.Unsubscribe(id1);
    pub.Notify("after-unsub");

    EXPECT_TRUE(r1.empty());
    ASSERT_EQ(r2.size(), 1u);
    EXPECT_EQ(r2[0], "after-unsub");
}

TEST(PublisherSubscriberTest, UnsubscribeNonExistentIsNoop) {
    PrintSection("UnsubscribeNonExistentIsNoop",
                 "Verify unsubscribing a bogus id does not crash or alter state.");

    Publisher<std::string> pub;
    pub.Subscribe([](const std::string&) {});
    pub.Unsubscribe(999);

    EXPECT_EQ(pub.SubscriberCount(), 1u);
}

TEST(PublisherSubscriberTest, NotifyWithNoSubscribers) {
    PrintSection("NotifyWithNoSubscribers",
                 "Verify notifying with zero subscribers does not crash.");

    Publisher<std::string> pub;
    pub.Notify("should not crash");  // No assertion -- just must not crash.
    EXPECT_EQ(pub.SubscriberCount(), 0u);
}

TEST(PublisherSubscriberTest, MultipleNotificationsInOrder) {
    PrintSection("MultipleNotificationsInOrder",
                 "Verify multiple notifications arrive in the order they were sent.");

    Publisher<std::string> pub;
    std::vector<std::string> received;

    pub.Subscribe([&received](const std::string& msg) {
        received.push_back(msg);
    });

    pub.Notify("a");
    pub.Notify("b");
    pub.Notify("c");

    ASSERT_EQ(received.size(), 3u);
    EXPECT_EQ(received[0], "a");
    EXPECT_EQ(received[1], "b");
    EXPECT_EQ(received[2], "c");
}

// ---------------------------------------------------------------------------
// Concurrent publish + subscribe with jthread
// ---------------------------------------------------------------------------

TEST(PublisherSubscriberTest, ConcurrentPublishAndSubscribe) {
    PrintSection("ConcurrentPublishAndSubscribe",
                 "Verify thread safety under concurrent Notify and Subscribe.");

    Publisher<int> pub;
    std::atomic<int> total_received{0};

    // Pre-subscribe a few listeners.
    constexpr int kInitialSubscribers = 4;
    for (int i = 0; i < kInitialSubscribers; ++i) {
        pub.Subscribe([&total_received](const int&) {
            total_received.fetch_add(1, std::memory_order_relaxed);
        });
    }

    constexpr int kNotifyCount = 500;
    constexpr int kSubscribeCount = 50;

    // Launch publisher threads (concurrent Notify).
    std::vector<std::jthread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&pub]() {
            for (int i = 0; i < kNotifyCount; ++i) {
                pub.Notify(i);
            }
        });
    }

    // Launch subscriber threads (concurrent Subscribe).
    std::mutex ids_mutex;
    std::vector<Publisher<int>::SubscriptionId> new_ids;
    for (int t = 0; t < 2; ++t) {
        threads.emplace_back([&pub, &total_received, &ids_mutex, &new_ids]() {
            for (int i = 0; i < kSubscribeCount; ++i) {
                auto id = pub.Subscribe([&total_received](const int&) {
                    total_received.fetch_add(1, std::memory_order_relaxed);
                });
                std::lock_guard lock(ids_mutex);
                new_ids.push_back(id);
            }
        });
    }

    // Wait for all threads to finish.
    threads.clear();

    // Sanity: we should have received a positive number of callbacks.
    EXPECT_GT(total_received.load(), 0);
    // And subscriber count should include the dynamically added ones.
    EXPECT_GE(pub.SubscriberCount(),
              static_cast<std::size_t>(kInitialSubscribers));
}

TEST(PublisherSubscriberTest, ConcurrentUnsubscribe) {
    PrintSection("ConcurrentUnsubscribe",
                 "Verify thread safety under concurrent Notify and Unsubscribe.");

    Publisher<int> pub;
    std::atomic<int> total_received{0};

    constexpr int kSubscribers = 100;
    std::vector<Publisher<int>::SubscriptionId> ids;
    for (int i = 0; i < kSubscribers; ++i) {
        ids.push_back(pub.Subscribe([&total_received](const int&) {
            total_received.fetch_add(1, std::memory_order_relaxed);
        }));
    }

    // Concurrently notify and unsubscribe.
    std::vector<std::jthread> threads;

    // Notifier threads.
    for (int t = 0; t < 2; ++t) {
        threads.emplace_back([&pub]() {
            for (int i = 0; i < 200; ++i) {
                pub.Notify(i);
            }
        });
    }

    // Unsubscriber threads.
    for (int t = 0; t < 2; ++t) {
        threads.emplace_back([&pub, &ids, t]() {
            int start = t * (kSubscribers / 2);
            int end = start + (kSubscribers / 2);
            for (int i = start; i < end; ++i) {
                pub.Unsubscribe(ids[static_cast<std::size_t>(i)]);
            }
        });
    }

    threads.clear();

    // All subscribers were unsubscribed.
    EXPECT_EQ(pub.SubscriberCount(), 0u);
    EXPECT_GT(total_received.load(), 0);
}

} // namespace
