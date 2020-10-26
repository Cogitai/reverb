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

#include "reverb/cc/local_table_sampler.h"

#include <algorithm>
#include <string>

#include "grpcpp/impl/codegen/client_context.h"
#include "absl/time/time.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/support/grpc_util.h"
#include "reverb/cc/table.h"
#include "reverb/cc/tensor_compression.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"

namespace deepmind {
namespace reverb {
namespace {

tensorflow::Status AsSample(const Table::SampledItem& si,
                            std::unique_ptr<Sample>* sample) {
  const auto& item = si.item;

  // Extract all chunks belonging to this sample.
  std::list<std::vector<tensorflow::Tensor>> chunks;

  // The chunks are not required to be aligned perfectly with the data so a
  // part of the first chunk is potentially stripped. The same applies to the
  // last part of the final chunk.
  int64_t offset = item.sequence_range().offset();
  int64_t remaining = item.sequence_range().length();

  for (auto& chunk : si.chunks) {
    REVERB_CHECK_GT(remaining, 0);

    std::vector<tensorflow::Tensor> batches;
    batches.reserve(chunk->data().data_size());

    int64_t batch_size = -1;

    // Note(piyushk): Original implementation was going back to front for
    // releasing protobuf memory, but we don't.
    for (int64_t insert_index = 0;
         insert_index < chunk->data().data_size(); ++insert_index) {
      tensorflow::Tensor batch = DecompressTensorFromProto(
          chunk->data().data(insert_index));
      if (chunk->data().delta_encoded()) {
        batch = DeltaEncode(batch, /*encode=*/false);
      }
      if (batch_size < 0) {
        batch_size = batch.dim_size(0);
      } else {
        if (batch_size != batch.dim_size(0)) {
          return tensorflow::errors::Internal(
              "Chunks of the same response must have identical batch size, but "
              "first chunk has batch size ",
              batch_size, " while the current chunk has batch size ",
              batch.dim_size(0));
        }
      }
      batch = batch.Slice(offset,
                          std::min<int64_t>(offset + remaining, batch_size));
      if (!batch.IsAligned()) {
        batch = tensorflow::tensor::DeepCopy(batch);
      }
      batches.push_back(std::move(batch));
    }

    chunks.push_back(std::move(batches));
    remaining -= std::min<int64_t>(remaining, batch_size - offset);
    offset = 0;
  }

  REVERB_CHECK_EQ(remaining, 0);
  *sample = absl::make_unique<deepmind::reverb::Sample>(
      si.item.key(), si.probability,
      si.table_size, si.item.priority(),
      std::move(chunks));

  return tensorflow::Status::OK();
}

}  // namespace

LocalTableSampler::LocalTableSampler(
        Table* table, const Options& options,
        internal::DtypesAndShapes dtypes_and_shapes)
    : table_(table),
      rate_limiter_timeout_(options.rate_limiter_timeout),
      active_sample_(nullptr),
      sampled_items_(),
      sampled_items_index_(0),
      batch_size_(
          options.flexible_batch_size == LocalTableSampler::kAutoSelectValue
          ? table_->DefaultFlexibleBatchSize()
          : options.flexible_batch_size),
      dtypes_and_shapes_(std::move(dtypes_and_shapes)) {
  REVERB_CHECK(options.flexible_batch_size == kAutoSelectValue ||
               options.flexible_batch_size > 0);
  REVERB_CHECK(options.rate_limiter_timeout >= absl::ZeroDuration());
}

LocalTableSampler::~LocalTableSampler() { Close(); }

tensorflow::Status LocalTableSampler::GetNextTimestep(
    std::vector<tensorflow::Tensor>* data, bool* end_of_sequence) {
  TF_RETURN_IF_ERROR(MaybeSampleNext());

  *data = active_sample_->GetNextTimestep();
  TF_RETURN_IF_ERROR(ValidateAgainstOutputSpec(*data, /*time_step=*/true));

  if (end_of_sequence != nullptr) {
    *end_of_sequence = active_sample_->is_end_of_sample();
  }

  return tensorflow::Status::OK();
}

tensorflow::Status LocalTableSampler::GetNextSample(
    std::vector<tensorflow::Tensor>* data) {
  std::unique_ptr<Sample> sample;
  TF_RETURN_IF_ERROR(PopNextSample(&sample));
  TF_RETURN_IF_ERROR(sample->AsBatchedTimesteps(data));
  TF_RETURN_IF_ERROR(ValidateAgainstOutputSpec(*data, /*time_step=*/false));
  return tensorflow::Status::OK();
}

tensorflow::Status LocalTableSampler::ValidateAgainstOutputSpec(
    const std::vector<tensorflow::Tensor>& data, bool time_step) {
  if (!dtypes_and_shapes_) {
    return tensorflow::Status::OK();
  }

  if (data.size() != dtypes_and_shapes_->size()) {
    return tensorflow::errors::InvalidArgument(
        "Inconsistent number of tensors received from table '", table_,
        "'.  Specification has ", dtypes_and_shapes_->size(),
        " tensors, but data coming from the table shows ", data.size(),
        " tensors.\nTable signature: ",
        internal::DtypesShapesString(*dtypes_and_shapes_),
        ".\nIncoming tensor signature: ",
        internal::DtypesShapesString(internal::SpecsFromTensors(data)));
  }

  for (int i = 0; i < data.size(); ++i) {
    tensorflow::TensorShape elem_shape;
    if (!time_step) {
      // Remove the outer dimension from data[i].shape() so we can properly
      // compare against the spec (which doesn't have the sequence dimension).
      elem_shape = data[i].shape();
      if (elem_shape.dims() == 0) {
        return tensorflow::errors::InvalidArgument(
            "Invalid tensor shape received from table '", table_,
            "'.  "
            "time_step is false but data[",
            i,
            "] has scalar shape "
            "(no time dimension).");
      }
      elem_shape.RemoveDim(0);
    }

    auto* shape_ptr = time_step ? &(data[i].shape()) : &elem_shape;
    if (data[i].dtype() != dtypes_and_shapes_->at(i).dtype ||
        !dtypes_and_shapes_->at(i).shape.IsCompatibleWith(*shape_ptr)) {
      return tensorflow::errors::InvalidArgument(
          "Received incompatible tensor at flattened index ", i,
          " from table '", table_, "'.  Specification has (dtype, shape): (",
          tensorflow::DataTypeString(dtypes_and_shapes_->at(i).dtype), ", ",
          dtypes_and_shapes_->at(i).shape.DebugString(),
          ").  Tensor has (dtype, shape): (",
          tensorflow::DataTypeString(data[i].dtype()), ", ",
          shape_ptr->DebugString(), ").\nTable signature: ",
          internal::DtypesShapesString(*dtypes_and_shapes_));
    }
  }
  return tensorflow::Status::OK();
}

void LocalTableSampler::Close() {
  {
    if (closed_) return;
    closed_ = true;
  }
}

tensorflow::Status LocalTableSampler::MaybeSampleNext() {
  if (active_sample_ != nullptr && !active_sample_->is_end_of_sample()) {
    return tensorflow::Status::OK();
  }

  return PopNextSample(&active_sample_);
}

tensorflow::Status LocalTableSampler::PopNextSample(
    std::unique_ptr<Sample>* sample) {
  if (closed_) {
    return tensorflow::errors::Cancelled(
        "LocalTableSampler has been cancelled.");
  }

  if (sampled_items_index_ == 0) {
    auto status = table_->SampleFlexibleBatch(
        &sampled_items_,
        batch_size_,
        rate_limiter_timeout_);
    if (!status.ok()) {
      return tensorflow::errors::Internal("Unable to fetch batch.");
    }
    sampled_items_index_ = sampled_items_.size();
  }

  return AsSample(
      sampled_items_[sampled_items_.size() - sampled_items_index_--], sample);
}

tensorflow::Status LocalTableSampler::Options::Validate() const {
  if (rate_limiter_timeout < absl::ZeroDuration()) {
    return tensorflow::errors::InvalidArgument("rate_limiter_timeout (",
                                               rate_limiter_timeout,
                                               ") must not be negative.");
  }
  if (flexible_batch_size < 1 && flexible_batch_size != kAutoSelectValue) {
    return tensorflow::errors::InvalidArgument(
        "flexible_batch_size (", flexible_batch_size, ") must be ",
        kAutoSelectValue, " or >= 1");
  }
  return tensorflow::Status::OK();
}

}  // namespace reverb
}  // namespace deepmind
