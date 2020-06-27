using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Util;
using static Tensorflow.tensorflow;

namespace Tensorflow.Gradients
{
    public interface ITape
    {
        void PopTape(ITape tape);

        bool ShouldRecord(long[] tensor_ids, TF_DataType[] dtypes);

        void RecordOperation(string op_type,
            Tensor[] input_tensors,
            TapeTensor[] output_tensors,
            long[] input_tensor_id,
            TF_DataType[] input_dtypes,
            Func<BackwardFunction> backward_function_getter);

        void VariableAccessed(ResourceVariable variable);

        void Watch(long tensor_id);

        ResourceVariable[] WatchedVariables();

        Tensor[] ComputeGradient(long[] target_tensor_ids,
            long[] source_tensor_ids,
            UnorderedMap<long, TapeTensor> sources_that_are_targets,
            Tensor[] output_gradients);
    }
}
