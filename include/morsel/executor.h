#pragma once

// Morsel-driven parallel execution (Leis et al., SIGMOD 2014).
//
// A plan is cut into pipelines at its breakers (hash-join builds,
// aggregations, sorts). Each pipeline runs on every worker: a worker takes the
// next morsel (a range of source rows) from a shared atomic counter, pushes it
// through the pipeline's operators one vector at a time, and adds the result to
// its own thread-local sink state. When the source is exhausted the sink merges
// the local states, in parallel where that pays off. Workers that finish early
// just take more morsels, so load stays balanced without any planning.

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "morsel/plan.h"
#include "morsel/storage.h"

namespace morsel {

// Fixed set of workers. run(fn) calls fn(worker_id) on every worker (the
// calling thread is worker 0) and returns when all have finished.
class ThreadPool {
 public:
  explicit ThreadPool(int threads);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return static_cast<int>(workers_.size()) + 1; }
  void run(const std::function<void(int)>& fn);

  // Calls fn(i) for every i in [0, n), spread over the workers.
  void parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn);

 private:
  void worker_loop(int id);

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable start_cv_, done_cv_;
  const std::function<void(int)>* job_ = nullptr;
  std::size_t generation_ = 0;
  int pending_ = 0;
  bool stop_ = false;
};

struct PipelineProfile {
  std::string name;
  double ms = 0;
};

class Executor {
 public:
  explicit Executor(int threads = 0);  // 0: one per hardware thread
  int threads() const { return pool_.size(); }

  MaterializedTable run(const Plan& plan);

  // Wall time of each pipeline in the last run, in execution order.
  const std::vector<PipelineProfile>& last_profile() const { return profile_; }

 private:
  ThreadPool pool_;
  std::vector<PipelineProfile> profile_;
};

// Rows per morsel: small enough for load balancing (a 6M-row scan has ~180
// morsels), large enough that taking one is negligible next to processing it.
constexpr std::size_t kMorselRows = 16 * 2048;

}  // namespace morsel
