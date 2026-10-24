#include <atomic>

#include "morsel/executor.h"

namespace morsel {

ThreadPool::ThreadPool(int threads) {
  if (threads < 1) threads = 1;
  for (int i = 1; i < threads; ++i) workers_.emplace_back([this, i] { worker_loop(i); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  start_cv_.notify_all();
  for (auto& w : workers_) w.join();
}

void ThreadPool::worker_loop(int id) {
  std::size_t seen = 0;
  for (;;) {
    const std::function<void(int)>* job;
    {
      std::unique_lock<std::mutex> lock(mu_);
      start_cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
      job = job_;
    }
    (*job)(id);
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (--pending_ == 0) done_cv_.notify_one();
    }
  }
}

void ThreadPool::run(const std::function<void(int)>& fn) {
  if (workers_.empty()) {
    fn(0);
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    job_ = &fn;
    pending_ = static_cast<int>(workers_.size());
    ++generation_;
  }
  start_cv_.notify_all();
  fn(0);
  std::unique_lock<std::mutex> lock(mu_);
  done_cv_.wait(lock, [&] { return pending_ == 0; });
}

void ThreadPool::parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn) {
  if (n == 0) return;
  if (n == 1 || workers_.empty()) {
    for (std::size_t i = 0; i < n; ++i) fn(i);
    return;
  }
  std::atomic<std::size_t> next{0};
  run([&](int) {
    for (std::size_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < n;) fn(i);
  });
}

}  // namespace morsel
