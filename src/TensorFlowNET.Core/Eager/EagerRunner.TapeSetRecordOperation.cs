using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Gradients;

namespace Tensorflow.Eager
{
    public partial class EagerRunner
    {
        public bool TapeSetRecordOperation(string op_type,
            Tensor[] input_tensors,
            Tensor[] output_tensors,
            BackwardFunction backward_function)
        {
            var output_info = output_tensors.Select(x => new TapeTensor(x)).ToArray();

            if (!TapeSetRecordForwardprop(op_type, input_tensors, output_info,
                    backward_function))
                return false;

            TapeSetRecordBackprop(op_type, input_tensors, output_info,
                backward_function);

            return true;
        }
    }
}
