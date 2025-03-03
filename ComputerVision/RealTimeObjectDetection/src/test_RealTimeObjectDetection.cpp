/**
 * @file test_RealTimeObjectDetection.cpp
 * @brief Unit tests for RealTimeObjectDetection components
 */

#include "Detector.h"
#include "KalmanTracker.h"
#include "ThreadSafeQueue.h"
#include <cassert>
#include <iostream>
#include <thread>

namespace {

void TestDetectionStruct() {
    Detection det(cv::Rect_<float>(10, 20, 30, 40), 5, 0.95f);
    assert(det.ClassId == 5);
    assert(det.Confidence > 0.9f);
    assert(det.Bbox.x == 10);
    std::cout << "  TestDetectionStruct PASSED\n";
}

void TestTrackedObjectInit() {
    TrackedObject obj(1, cv::Rect_<float>(0, 0, 50, 50), 0, 0.9f);
    assert(obj.Id == 1);
    assert(obj.Age == 1);
    assert(obj.TotalVisibleCount == 1);
    assert(obj.ConsecutiveInvisibleCount == 0);
    std::cout << "  TestTrackedObjectInit PASSED\n";
}

void TestKalmanTrackerEmptyUpdate() {
    KalmanTracker tracker;
    std::vector<Detection> empty;
    auto tracks = tracker.Update(empty, 640, 480);
    assert(tracks.empty());
    assert(tracker.GetTrackCount() == 0);
    std::cout << "  TestKalmanTrackerEmptyUpdate PASSED\n";
}

void TestKalmanTrackerNewTrack() {
    KalmanTracker tracker(20, 1, 0.3f);  // minHits=1 so tracks confirm immediately
    std::vector<Detection> dets;
    dets.emplace_back(cv::Rect_<float>(100, 100, 50, 50), 0, 0.9f);

    auto tracks = tracker.Update(dets, 640, 480);
    assert(tracker.GetTrackCount() == 1);
    std::cout << "  TestKalmanTrackerNewTrack PASSED\n";
}

void TestThreadSafeQueuePushPop() {
    ThreadSafeQueue<int> queue;
    queue.Push(42);
    queue.Push(99);
    assert(queue.size() == 2);
    auto val = queue.TryPop();
    assert(val.has_value());
    assert(*val == 42);
    val = queue.TryPop();
    assert(*val == 99);
    assert(queue.empty());
    std::cout << "  TestThreadSafeQueuePushPop PASSED\n";
}

void TestThreadSafeQueueDone() {
    ThreadSafeQueue<int> queue;
    queue.Push(1);
    assert(!queue.IsDone());
    queue.Done();
    auto val = queue.TryPop();
    assert(val.has_value() && *val == 1);
    assert(queue.IsDone());
    std::cout << "  TestThreadSafeQueueDone PASSED\n";
}

}  // namespace

int main() {
    std::cout << "Running RealTimeObjectDetection tests...\n";
    TestDetectionStruct();
    TestTrackedObjectInit();
    TestKalmanTrackerEmptyUpdate();
    TestKalmanTrackerNewTrack();
    TestThreadSafeQueuePushPop();
    TestThreadSafeQueueDone();
    std::cout << "All RealTimeObjectDetection tests PASSED.\n";
    return 0;
}
