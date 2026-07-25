// Minimal persistent worker pool used by the CPU renderer.
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace sol {

class ThreadPool {
public:
    explicit ThreadPool(int threadCount = 0) {
        if (threadCount <= 0) threadCount = int(std::thread::hardware_concurrency());
        if (threadCount <= 0) threadCount = 4;
        threadCount_ = threadCount;
        workers_.reserve(size_t(threadCount));
        for (int i = 0; i < threadCount; ++i) workers_.emplace_back([this, i] { workerLoop(i); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (std::thread& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int threadCount() const { return threadCount_; }

    // Runs body(index, threadId) for index in [0, count) and blocks until done.
    void parallelFor(int count, const std::function<void(int, int)>& body) {
        if (count <= 0) return;
        if (threadCount_ <= 1) {
            for (int i = 0; i < count; ++i) body(i, 0);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            body_ = &body;
            nextIndex_.store(0, std::memory_order_relaxed);
            totalCount_ = count;
            activeWorkers_.store(threadCount_, std::memory_order_relaxed);
            ++generation_;
        }
        cv_.notify_all();
        runChunks(0);
        // Wait for the workers of this generation to drain.
        std::unique_lock<std::mutex> lock(mutex_);
        doneCv_.wait(lock, [this] { return activeWorkers_.load(std::memory_order_acquire) == 0; });
        body_ = nullptr;
        totalCount_ = 0;
    }

private:
    void runChunks(int threadId) {
        const std::function<void(int, int)>* body = body_;
        if (!body) return;
        for (;;) {
            const int index = nextIndex_.fetch_add(1, std::memory_order_relaxed);
            if (index >= totalCount_) break;
            (*body)(index, threadId);
        }
    }

    void workerLoop(int threadId) {
        uint64_t seenGeneration = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return stop_ || generation_ != seenGeneration; });
                if (stop_) return;
                seenGeneration = generation_;
            }
            runChunks(threadId + 1);
            if (activeWorkers_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(mutex_);
                doneCv_.notify_all();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable doneCv_;
    const std::function<void(int, int)>* body_ = nullptr;
    std::atomic<int> nextIndex_{0};
    std::atomic<int> activeWorkers_{0};
    int totalCount_ = 0;
    uint64_t generation_ = 0;
    bool stop_ = false;
    int threadCount_ = 1;
};

}  // namespace sol
