using System;
using System.Collections.Generic;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace Tensorflow.Gradients
{
    public partial class Tape
    {
        long next_op_id_ = 0;
        UnorderedMap<long, long> tensor_usage_;

        public void RecordOperation(string op_type,
            TapeTensor[] output_tensors,
            long[] input_tensor_id,
            TF_DataType[] input_dtypes,
            BackwardFunction backward_function)
        {
            if (!ShouldRecord(input_tensor_id, input_dtypes))
                return;

            foreach (var i in input_tensor_id)
            {
                tensor_usage_[i]++;
            }
            long op_id = next_op_id_++;
            
            foreach (var o in output_tensors)
            {
                tf.Logger.Debug($"RecordOperation: tensor_tape_[{o.GetID()}] = {op_id}");
                tensor_tape_[o.GetID()] = op_id;
                tensor_usage_[o.GetID()] = 1;
            }

            op_tape_[op_id] = new OpTapeEntry
            {
                op_type = op_type,
                output_tensor_info = output_tensors.ToArray(),
                input_tensor_id = input_tensor_id.ToArray(),
                backward_function = backward_function
            };
        }

        public void RecordOperation(string op_type,
            Tensor[] outputs,
            Tensor[] inputs,
            BackwardFunction backward_function)
        {
            tf.Runner.TFE_TapeSetRecordOperation(op_type, outputs, inputs, backward_function);
        }
    }
}
