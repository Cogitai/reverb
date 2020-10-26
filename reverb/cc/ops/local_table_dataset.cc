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

#include <tensorflow/core/framework/tensor_util.h>
#include "tensorflow/core/framework/dataset.h"

#include "reverb/cc/local_table_sampler.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/local_table_sampler.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "reverb/cc/tensor_compression.h"

namespace deepmind {
namespace reverb {
namespace {

using ::tensorflow::errors::Cancelled;
using ::tensorflow::errors::FailedPrecondition;
using ::tensorflow::errors::InvalidArgument;
using ::tensorflow::errors::Unimplemented;

REGISTER_OP("ReverbLocalTableDataset")
    .Input("table: int64")
    .Attr("sequence_length: int = -1")
    .Attr("emit_timesteps: bool = true")
    .Attr("flexible_batch_size: int = -1")
    .Attr("rate_limiter_timeout_ms: int = -1")
    .Attr("dtypes: list(type) >= 1")
    .Attr("shapes: list(shape) >= 1")
    .Output("dataset: variant")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Stream samples from table `table`. The pointer to the table is passed in as
 an int64.

`dtypes` and `shapes` must match the type and shape of a single "timestep"
within sampled sequences. That is, (key, priority, table_size, ...data passed to
`Writer::Append` at insertion time). This is the type and shape of
tensors returned by `GetNextTimestep`.

sequence_length: (Defaults to -1, i.e unknown) The number of timesteps in
the samples. If set then the length of the received samples are checked against
this value.

`emit_timesteps` (defaults to true) determines whether individual timesteps or
complete sequences should be returned from the iterators. When set to false
(i.e return sequences), `shapes` must have dim[0] equal to `sequence_length`.
Emitting complete samples is more efficient as it avoids the memcopies involved
in splitting up a sequence and then batching it up again.

`flexible_batch_size` (defaults to -1, i.e auto selected) is the maximum number
of items to sampled from `Table` with a single call. Values > 1 enables
`Table::SampleFlexibleBatch` to return more than one item (but no more than
`flexible_batch_size`) in a single call without releasing the table lock iff
the rate limiter allows it.

`rate_limiter_timeout_ms` (defaults to -1, i.e. never time out) is the number of
milliseconds an iterator should wait for new data from the sampler before timing
out. This can be useful, e.g., when the Reverb server receives data in
collection stages - and a dataset iterator should stop when no new data is
available for a while. If `rate_limiter_timeout_ms >= 0`, an iterator that waits
for data longer than this will close and mark the input sequence as finished.
Note that the timeout behavior depends on the Table's rate limiter. For example,
the table may contain data, but the rate limiter may pause sampling - and this
can cause a timeout to occur.
)doc");

class ReverbLocalTableDatasetOp : public tensorflow::data::DatasetOpKernel {
 public:
  explicit ReverbLocalTableDatasetOp(tensorflow::OpKernelConstruction* ctx)
      : tensorflow::data::DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sequence_length", &sequence_length_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("emit_timesteps", &emit_timesteps_));
    tensorflow::int64 rate_limiter_timeout_ms;
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("flexible_batch_size",
                          &sampler_options_.flexible_batch_size));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("rate_limiter_timeout_ms", &rate_limiter_timeout_ms));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shapes", &shapes_));

    sampler_options_.rate_limiter_timeout =
        Int64MillisToNonnegativeDuration(rate_limiter_timeout_ms);

    if (!emit_timesteps_) {
      for (int i = 0; i < shapes_.size(); i++) {
        OP_REQUIRES(ctx, shapes_[i].dims() != 0,
                    InvalidArgument(
                        "When emit_timesteps is false, all elements of shapes "
                        "must have "
                        "dim[0] = sequence_length (",
                        sequence_length_, "). Element ", i,
                        " of flattened shapes has rank 0 and thus no dim[0]."));

        OP_REQUIRES(ctx, shapes_[i].dim_size(0) == sequence_length_,
                    InvalidArgument("When emit_timesteps is false, all "
                                    "elements of shapes must have "
                                    "dim[0] = sequence_length (",
                                    sequence_length_, "). Element ", i,
                                    " of flattened shapes has dim[0] = ",
                                    shapes_[i].dim_size(0), "."));
      }
    }
  }

  void MakeDataset(tensorflow::OpKernelContext* ctx,
                   tensorflow::data::DatasetBase** output) override {
    tensorflow::int64 table;
    OP_REQUIRES_OK(ctx,
                   tensorflow::data::ParseScalarArgument<tensorflow::int64>(
                           ctx, "table", &table));

    *output = new Dataset(ctx, reinterpret_cast<Table*>(table),
                          dtypes_, shapes_, sampler_options_,
                          sequence_length_, emit_timesteps_);
  }

 private:
  class Dataset : public tensorflow::data::DatasetBase {
   public:
    Dataset(tensorflow::OpKernelContext* ctx, Table* table,
            tensorflow::DataTypeVector dtypes,
            std::vector<tensorflow::PartialTensorShape> shapes,
            const LocalTableSampler::Options& sampler_options,
            int sequence_length, bool emit_timesteps)
        : tensorflow::data::DatasetBase(tensorflow::data::DatasetContext(ctx)),
          table_(table),
          dtypes_(std::move(dtypes)),
          shapes_(std::move(shapes)),
          sampler_options_(sampler_options),
          sequence_length_(sequence_length),
          emit_timesteps_(emit_timesteps) {}

    std::unique_ptr<tensorflow::data::IteratorBase> MakeIteratorInternal(
        const std::string& prefix) const override {
      return absl::make_unique<Iterator>(
          tensorflow::data::DatasetIterator<Dataset>::Params{
              this, absl::StrCat(prefix, "::ReverbLocalTableDataset")},
          table_, sampler_options_, sequence_length_,
          emit_timesteps_, dtypes_, shapes_);
    }

    const tensorflow::DataTypeVector& output_dtypes() const override {
      return dtypes_;
    }

    const std::vector<tensorflow::PartialTensorShape>& output_shapes()
        const override {
      return shapes_;
    }

    std::string DebugString() const override {
      return "ReverbDatasetOp::Dataset";
    }

    tensorflow::Status CheckExternalState() const override {
      return FailedPrecondition(DebugString(), " depends on external state.");
    }

   protected:
    tensorflow::Status AsGraphDefInternal(
        tensorflow::data::SerializationContext* ctx, DatasetGraphDefBuilder* b,
        tensorflow::Node** output) const override {
      tensorflow::AttrValue sequence_length_attr;
      tensorflow::AttrValue emit_timesteps_attr;
      tensorflow::AttrValue flexible_batch_size_attr;
      tensorflow::AttrValue rate_limiter_timeout_ms_attr;
      tensorflow::AttrValue dtypes_attr;
      tensorflow::AttrValue shapes_attr;

      tensorflow::Node* table = nullptr;

      TF_RETURN_IF_ERROR(
              b->AddScalar<tensorflow::int64>((int64_t)table_, &table));

      b->BuildAttrValue(sequence_length_, &sequence_length_attr);
      b->BuildAttrValue(emit_timesteps_, &emit_timesteps_attr);
      b->BuildAttrValue(sampler_options_.flexible_batch_size,
          &flexible_batch_size_attr);
      b->BuildAttrValue(
          static_cast<tensorflow::int64>(NonnegativeDurationToInt64Millis(
              sampler_options_.rate_limiter_timeout)),
          &rate_limiter_timeout_ms_attr);
      b->BuildAttrValue(dtypes_, &dtypes_attr);
      b->BuildAttrValue(shapes_, &shapes_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(
          this,
          /*inputs=*/{table},
          /*attrs=*/
          {
              {"sequence_length", sequence_length_attr},
              {"emit_timesteps", emit_timesteps_attr},
              {"flexible_batch_size", flexible_batch_size_attr},
              {"rate_limiter_timeout_ms", rate_limiter_timeout_ms_attr},
              {"dtypes", dtypes_attr},
              {"shapes", shapes_attr},
          },
          output));

      return tensorflow::Status::OK();
    }

   private:
    class Iterator : public tensorflow::data::DatasetIterator<Dataset> {
     public:
      explicit Iterator(
          const Params& params, Table* table,
          const LocalTableSampler::Options& sampler_options,
          int sequence_length, bool emit_timesteps,
          const tensorflow::DataTypeVector& dtypes,
          const std::vector<tensorflow::PartialTensorShape>& shapes)
          : DatasetIterator<Dataset>(params),
            table_(table),
            sampler_options_(sampler_options),
            sequence_length_(sequence_length),
            emit_timesteps_(emit_timesteps),
            dtypes_(dtypes),
            shapes_(shapes),
            step_within_sample_(0) { }

        tensorflow::Status Initialize(
          tensorflow::data::IteratorContext* ctx) override {
        // If sequences are emitted then the all shapes will start with the
        // sequence length. The validation expects the shapes of a single
        // timestep so if sequences are emitted then we need to trim the leading
        // dim on all shapes before validating it.
        auto validation_shapes = shapes_;
        if (!emit_timesteps_) {
          for (auto& shape : validation_shapes) {
            shape.RemoveDim(0);
          }
        }

        TF_RETURN_IF_ERROR(sampler_options_.Validate());
        // TODO(piyushk): Note that dtypes and shapes validation still needs
        // to be added.
        sampler_ = absl::make_unique<LocalTableSampler>(
            table_, sampler_options_);

        return tensorflow::Status::OK();
      }

      tensorflow::Status GetNextInternal(
          tensorflow::data::IteratorContext* ctx,
          std::vector<tensorflow::Tensor>* out_tensors,
          bool* end_of_sequence) override {
        REVERB_CHECK(sampler_.get() != nullptr) << "Initialize was not called?";

        tensorflow::Status status;
        if (emit_timesteps_) {
          bool last_timestep = false;
          status = sampler_->GetNextTimestep(out_tensors, &last_timestep);

          step_within_sample_++;

          if (last_timestep && sequence_length_ > 0 &&
              step_within_sample_ != sequence_length_) {
            return InvalidArgument(
                "Received sequence of invalid length. Expected ",
                sequence_length_, " steps, got ", step_within_sample_);
          }
          if (step_within_sample_ == sequence_length_ && !last_timestep) {
            return InvalidArgument(
                "Receieved sequence did not terminate after expected number of "
                "steps (",
                sequence_length_, ").");
          }
          if (last_timestep) {
            step_within_sample_ = 0;
          }
        } else {
          status = sampler_->GetNextSample(out_tensors);
        }

        if (status.ok()) {
          *end_of_sequence = false;
          return status;
        } else if (tensorflow::errors::IsDeadlineExceeded(status) &&
                   sampler_options_.rate_limiter_timeout <
                       absl::InfiniteDuration() &&
                   status.error_message().find("Rate Limiter") !=
                       std::string::npos) {
          // TODO(157580783): Move the error string above to a common library.
          *end_of_sequence = true;
          return tensorflow::Status::OK();
        } else {
          return status;
        }
      }

     protected:
      tensorflow::Status SaveInternal(
          tensorflow::data::SerializationContext* ctx,
          tensorflow::data::IteratorStateWriter* writer) override {
        return Unimplemented("SaveInternal is currently not supported");
      }

      tensorflow::Status RestoreInternal(
          tensorflow::data::IteratorContext* ctx,
          tensorflow::data::IteratorStateReader* reader) override {
        return Unimplemented("RestoreInternal is currently not supported");
      }

     private:
      Table* table_;
      const LocalTableSampler::Options sampler_options_;
      const int sequence_length_;
      const bool emit_timesteps_;
      const tensorflow::DataTypeVector& dtypes_;
      const std::vector<tensorflow::PartialTensorShape>& shapes_;
      std::unique_ptr<LocalTableSampler> sampler_;
      int step_within_sample_;
    };  // Iterator.

    Table* table_;
    const tensorflow::DataTypeVector dtypes_;
    const std::vector<tensorflow::PartialTensorShape> shapes_;
    const LocalTableSampler::Options sampler_options_;
    const int sequence_length_;
    const bool emit_timesteps_;
  };  // Dataset.

  LocalTableSampler::Options sampler_options_;
  int sequence_length_;
  bool emit_timesteps_;
  tensorflow::DataTypeVector dtypes_;
  std::vector<tensorflow::PartialTensorShape> shapes_;

  TF_DISALLOW_COPY_AND_ASSIGN(ReverbLocalTableDatasetOp);
};

REGISTER_KERNEL_BUILDER(
    Name("ReverbLocalTableDataset").Device(tensorflow::DEVICE_CPU),
    ReverbLocalTableDatasetOp
);

}  // namespace
}  // namespace reverb
}  // namespace deepmind
