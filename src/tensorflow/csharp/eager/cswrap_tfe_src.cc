#include <cstring>
#include <thread>

#include "tensorflow/core/lib/core/errors.h"
#include "cswrap_tfe.h"

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"

#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"


namespace {

void RecordGradient(const char* op_name, void* inputs,
                                void* attrs, void* results,
                                const char* name) {
  // std::vector<tensorflow::int64> input_ids = MakeTensorIDList(inputs);
  
}

}