// Copyright 2019 DeepMind Technologies Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LEARNING_DEEPMIND_REPLAY_REVERB_LOCAL_TABLE_SAMPLER_H_
#define LEARNING_DEEPMIND_REPLAY_REVERB_LOCAL_TABLE_SAMPLER_H_

#include <stddef.h>

#include <list>
#include <memory>
#include <string>
#include <vector>

#include <cstdint>
#include "absl/time/time.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/sampler.h"
#include "reverb/cc/support/queue.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/table.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace deepmind {
namespace reverb {

// The `LocalTableSampler` class should be used to retrieve samples from a
// table object directly.
//
// Concurrent calls to a single `LocalTableSampler` object are NOT supported!
//
class LocalTableSampler {
 public:
  static const int kAutoSelectValue = -1;

  struct Options {
    // `rate_limiter_timeout` is the timeout that workers will use when waiting
    // for samples. This timeout is passed directly to the
    // `Table::Sample()` call on the server. When a timeout occurs, the Sample
    // status of `DeadlineExceeded` is returned.
    //
    // The default is to wait forever - or until the connection closes, or
    // `Close` is called, whichever comes first.
    absl::Duration rate_limiter_timeout = absl::InfiniteDuration();

    // The maximum number of items to sampled from `Table` with single call.
    // Values > 1 enables `Table::SampleFlexibleBatch` to return more than one
    // item (but no more than `flexible_batch_size`) in a single call without
    // releasing the table lock iff the rate limiter allows it.
    //
    // When set to `kAutoSelectValue`, `kDefaultFlexibleBatchSize` is used.
    int flexible_batch_size = kAutoSelectValue;

    // Checks that field values are valid and returns `InvalidArgument` if any
    // field value invalid.
    tensorflow::Status Validate() const;
  };

  // Constructs a new `LocalTableSampler`.
  //
  // `table` is the `Table` to sample from.
  // `options` defines details of how to samples.
  // `dtypes_and_shapes` describes the output signature (if any) to expect.
  LocalTableSampler(Table* table, const Options& options,
          internal::DtypesAndShapes dtypes_and_shapes = absl::nullopt);

  // Joins worker threads through call to `Close`.
  virtual ~LocalTableSampler();

  // Blocks until a timestep has been retrieved or until a non transient error
  // is encountered or `Close` has been called.
  tensorflow::Status GetNextTimestep(std::vector<tensorflow::Tensor>* data,
                                     bool* end_of_sequence);

  // Blocks until a complete sample has been retrieved or until a non transient
  // error is encountered or `Close` has been called.
  tensorflow::Status GetNextSample(std::vector<tensorflow::Tensor>* data);

  // Closes sthe sampler. Any blocking or future call to `GetNextTimestep` or
  // `GetNextSample` will return CancelledError without blocking.
  void Close();

  // LocalTableSampler is neither copyable nor movable.
  LocalTableSampler(const LocalTableSampler&) = delete;
  LocalTableSampler& operator=(const LocalTableSampler&) = delete;

 private:
  // Validates the `data` vector against `dtypes_and_shapes`.
  //
  // `data` is the data received by GetNextTimeStep or GetNextSample.
  // `time_step` is `true` if `GetNextTimeStep` is the caller and `false`
  //   if `GetNextSample` is the caller.
  tensorflow::Status ValidateAgainstOutputSpec(
      const std::vector<tensorflow::Tensor>& data, bool time_step);

  // If `active_sample_` has been read, blocks until a sample has been retrieved
  // (popped from `samples_`) and populates `active_sample_`.
  tensorflow::Status MaybeSampleNext();

  // Blocks until a complete sample has been retrieved or until a non transient
  // error is encountered.
  tensorflow::Status PopNextSample(std::unique_ptr<Sample>* sample);

  // Table we are sampling from.
  Table* table_;

  // The rate limiter timeout argument that all workers pass to SampleStream.
  const absl::Duration rate_limiter_timeout_;

  // Remaining timesteps of the currently active sample.
  std::unique_ptr<Sample> active_sample_;

  // We sample Table::SampledItem in batches. sampled_items_ contains the
  // the batch of items that has been fetched currently,
  // and sampled_items_index_ keeps track of how many items in that batch have
  // been returned.
  std::vector<Table::SampledItem> sampled_items_;
  int sampled_items_index_ = 0;
  int batch_size_ = 1;

  // The dtypes and shapes users expect from either `GetNextTimestep` or
  // `GetNextSample` (whichever they plan to call).  May be absl::nullopt,
  // meaning unknown.
  const internal::DtypesAndShapes dtypes_and_shapes_;

  // Set if `Close` called.
  bool closed_ = false;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // LEARNING_DEEPMIND_REPLAY_REVERB_LOCAL_TABLE_SAMPLER_H_
