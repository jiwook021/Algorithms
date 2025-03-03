/**
 * @file PublisherSubscriber.hpp
 * @brief Publisher-Subscriber pattern -- callback-based, thread-safe, templatized.
 *
 * @details
 * The Publisher maintains a list of callback-based subscriptions. When an event
 * occurs, the Publisher calls Notify(), which broadcasts the message to every
 * registered callback. Thread safety is provided by std::shared_mutex:
 * concurrent Notify() calls share the lock, while Subscribe()/Unsubscribe()
 * take an exclusive lock.
 *
 * Usage:
 *   1. Create a Publisher<MessageType>.
 *   2. Subscribe with a callback; receive a SubscriptionId.
 *   3. Call publisher.Notify(message) -- all subscribers receive it.
 *   4. Call publisher.Unsubscribe(id) to stop receiving events.
 *
 * Complexity:
 *   - Subscribe():   O(1) amortized
 *   - Unsubscribe(): O(n)
 *   - Notify():      O(n)
 */

#ifndef PUBLISHER_SUBSCRIBER_HPP
#define PUBLISHER_SUBSCRIBER_HPP

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <vector>

/**
 * @class Publisher
 * @brief Maintains callback-based subscriptions and broadcasts notifications.
 * @tparam MessageType The type of message to publish. Must be copy-constructible.
 *
 * Thread safety: Notify() acquires a shared lock so multiple threads can publish
 * concurrently. Subscribe()/Unsubscribe() acquire an exclusive lock.
 */
template<typename MessageType>
    requires std::copy_constructible<MessageType>
class Publisher {
public:
    using Callback = std::function<void(const MessageType&)>;
    using SubscriptionId = std::size_t;

    /**
     * @brief Register a callback to receive future notifications.
     * @param callback The function to invoke on each notification.
     * @return A unique SubscriptionId for later unsubscription.
     */
    SubscriptionId Subscribe(Callback callback) {
        std::unique_lock lock(mutex_);
        SubscriptionId id = next_id_++;
        subscribers_.push_back({id, std::move(callback)});
        return id;
    }

    /**
     * @brief Remove a subscription by its id. No-op if not found.
     * @param id The SubscriptionId returned by Subscribe().
     */
    void Unsubscribe(SubscriptionId id) {
        std::unique_lock lock(mutex_);
        auto it = std::find_if(subscribers_.begin(), subscribers_.end(),
                               [id](const Subscription& s) { return s.id == id; });
        if (it != subscribers_.end()) {
            subscribers_.erase(it);
        }
    }

    /**
     * @brief Send a message to every registered subscriber.
     * @param message The notification payload.
     *
     * Uses a shared lock so multiple threads can call Notify() concurrently.
     */
    void Notify(const MessageType& message) {
        std::shared_lock lock(mutex_);
        for (const auto& sub : subscribers_) {
            sub.callback(message);
        }
    }

    /**
     * @brief Return the number of currently registered subscribers.
     * @return Subscriber count.
     */
    std::size_t SubscriberCount() const {
        std::shared_lock lock(mutex_);
        return subscribers_.size();
    }

private:
    /** @brief Internal subscription record binding an id to a callback. */
    struct Subscription {
        SubscriptionId id;
        Callback callback;
    };

    std::vector<Subscription> subscribers_;  ///< Registered subscriptions.
    mutable std::shared_mutex mutex_;        ///< Protects subscribers_.
    SubscriptionId next_id_{0};              ///< Monotonically increasing id counter.
};

#endif // PUBLISHER_SUBSCRIBER_HPP
