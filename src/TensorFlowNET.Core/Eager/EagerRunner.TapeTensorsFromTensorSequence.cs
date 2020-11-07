using System.Collections.Generic;
using Tensorflow.Gradients;

namespace Tensorflow.Eager
{
    public partial class EagerRunner
    {
        bool TapeTensorsFromTensorSequence(Tensor[] output_seq,
            List<TapeTensor> output_info)
        {
            for (var i = 0; i < output_seq.Length; ++i)
            {
                output_info.Add(TapeTensorFromTensor(output_seq[i]));
            }
            return true;
        }
    }
}
