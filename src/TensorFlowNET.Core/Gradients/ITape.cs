using System;
using Tensorflow.Util;

namespace Tensorflow.Gradients
{
    public interface ITape
    {
        void SetTapeId(int id);
        bool ShouldRecord(Tensor[] tensors);
        void StartRecord();
        void StopRecord();
        bool Persistent { get; }
        void RecordOperation(string op_type,
            Tensor[] input_tensors,
            TapeTensor[] output_tensors,
            Func<BackwardFunction> backward_function_getter);

        void VariableAccessed(ResourceVariable variable);

        void Watch(Tensor x);

        ResourceVariable[] WatchedVariables();

        Tensor[] ComputeGradient(Tensor[] target_tensor_ids,
            Tensor[] source_tensor_ids,
            UnorderedMap<Tensor, TapeTensor> sources_that_are_targets,
            Tensor[] output_gradients);
    }
}
