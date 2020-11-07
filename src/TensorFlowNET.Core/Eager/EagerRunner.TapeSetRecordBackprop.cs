using System;
using Tensorflow.Gradients;
using static Tensorflow.Binding;
using static Tensorflow.tensorflow;

namespace Tensorflow.Eager
{
    public partial class EagerRunner
    {
        void TapeSetRecordBackprop(string op_type,
            Tensor[] input_tensors,
            TapeTensor[] output_tensors,
            long[] input_ids,
            TF_DataType[] input_dtypes,
            Func<BackwardFunction> backward_function_getter)
        {
            if (!CouldBackprop())
            {
                return;
            }

            foreach (var tape in tf.GetTapeSet())
            {
                tape.RecordOperation(op_type, input_tensors, output_tensors,
                    input_ids, input_dtypes,
                    backward_function_getter);
            }
        }
    }
}
