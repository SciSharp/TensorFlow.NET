using System;
using System.Collections.Generic;
using Tensorflow.Gradients;
using static Tensorflow.tensorflow;

namespace Tensorflow.Eager
{
    public partial class EagerRunner
    {
        public bool TapeSetRecordOperation(string op_type,
            Tensor[] input_tensors,
            Tensor[] output_tensors,
            Func<BackwardFunction> backward_function_getter)
        {
            var output_info = new List<TapeTensor>();

            if (!TapeTensorsFromTensorSequence(output_tensors, output_info))
                return false;

            if (!TapeSetRecordForwardprop(op_type, input_tensors, output_info.ToArray(),
                    backward_function_getter))
                return false;

            TapeSetRecordBackprop(op_type, input_tensors, output_info.ToArray(),
                backward_function_getter);

            return true;
        }
    }
}
