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
            long[] input_ids,
            TF_DataType[] input_dtypes,
            BackwardFunction backward_function)
        {
            var output_info = output_tensors.Select(t => TapeTensorFromTensor(t)).ToArray();
            if (!TapeSetRecordForwardprop(op_type, input_tensors, output_info,
                    backward_function))
                return false;

            TapeSetRecordBackprop(op_type, output_info, input_ids, input_dtypes,
                backward_function);

            return true;
        }

        public void TFE_TapeSetRecordOperation(string op_type, Tensor[] output_tensors,
            Tensor[] input_tensors, BackwardFunction backward_function)
        {
            var input_ids = MakeTensorIDList(input_tensors);
            var input_dtypes = MakeTensorDtypeList(input_tensors);
            TapeSetRecordOperation(op_type, input_tensors, output_tensors, input_ids, input_dtypes,
                backward_function);
        }
    }
}
