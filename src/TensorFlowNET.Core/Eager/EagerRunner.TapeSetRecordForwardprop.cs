using System;
using Tensorflow.Gradients;
using static Tensorflow.tensorflow;

namespace Tensorflow.Eager
{
    public partial class EagerRunner
    {
        bool TapeSetRecordForwardprop(string op_type,
            Tensor[] input_tensors,
            TapeTensor[] output_tensors,
            long[] input_ids,
            TF_DataType[] input_dtypes,
            Func<BackwardFunction> backward_function_getter)
        {
            if (!CouldForwardprop())
            {
                return true;
            }

            throw new NotImplementedException("");
        }
    }
}
