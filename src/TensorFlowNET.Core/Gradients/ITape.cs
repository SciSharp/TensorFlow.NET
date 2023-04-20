using System;
using Tensorflow.Util;

namespace Tensorflow.Gradients
{
    public interface ITape
    {
        void SetTapeId(int id);
        bool ShouldRecord(long[] tensor_ids, TF_DataType[] tensor_dtypes);
        void StartRecord();
        void StopRecord();
        bool Persistent { get; }
        void RecordOperation(string op_type,
            TapeTensor[] output_tensors,
            long[] input_tensor_id,
            TF_DataType[] input_dtypes,
            BackwardFunction backward_function);

        void RecordOperation(string op_type,
            Tensor[] outputs,
            Tensor[] inputs,
            BackwardFunction backward_function);

        void VariableAccessed(IVariableV1 variable);

        void Watch(Tensor x);

        IVariableV1[] WatchedVariables();

        Tensor[] ComputeGradient(long[] target_tensor_ids,
            long[] source_tensor_ids,
            UnorderedMap<long, TapeTensor> sources_that_are_targets,
            List<Tensor> output_gradients,
            bool build_default_zeros_grads);
    }
}
