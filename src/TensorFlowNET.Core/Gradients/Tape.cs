using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Util;

namespace Tensorflow.Gradients
{
    public class Tape : ITape
    {
        public Tape(bool persistent, bool watch_accessed_variables)
        {

        }

        public Tensor[] ComputeGradient(long[] target_tensor_ids, long[] source_tensor_ids, UnorderedMap<long, TapeTensor> sources_that_are_targets, Tensor[] output_gradients)
        {
            throw new NotImplementedException();
        }

        public void PopTape(ITape tape)
        {
            throw new NotImplementedException();
        }

        public void RecordOperation(string op_type, Tensor[] input_tensors, TapeTensor[] output_tensors, long[] input_tensor_id, TF_DataType[] input_dtypes, Func<tensorflow.BackwardFunction> backward_function_getter)
        {
            throw new NotImplementedException();
        }

        public bool ShouldRecord(long[] tensor_ids, TF_DataType[] dtypes)
        {
            throw new NotImplementedException();
        }

        public void VariableAccessed(ResourceVariable variable)
        {
            throw new NotImplementedException();
        }

        public void Watch(long tensor_id)
        {
            throw new NotImplementedException();
        }

        public ResourceVariable[] WatchedVariables()
        {
            throw new NotImplementedException();
        }
    }
}
