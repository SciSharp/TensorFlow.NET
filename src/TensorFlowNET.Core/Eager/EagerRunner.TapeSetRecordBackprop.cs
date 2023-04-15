using System;
using Tensorflow.Gradients;
using static Tensorflow.Binding;

namespace Tensorflow.Eager
{
    public partial class EagerRunner
    {
        void TapeSetRecordBackprop(string op_type,
            TapeTensor[] output_info,
            long[] input_ids,
            TF_DataType[] input_detyps,
            BackwardFunction backward_function)
        {
            if (!CouldBackprop())
            {
                return;
            }

            foreach (var tape in tf.GetTapeSet())
            {
                tape.RecordOperation(op_type, output_info, input_ids, input_detyps, backward_function);
            }
        }
    }
}
