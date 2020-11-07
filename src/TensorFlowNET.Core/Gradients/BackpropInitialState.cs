using Tensorflow.Util;
using static Tensorflow.tensorflow;

namespace Tensorflow.Gradients
{
    public class BackpropInitialState
    {
        public OpTape<BackwardFunction, TapeTensor> op_tape { get; set; }
        /// <summary>
        /// Map from tensor ID to how many references still exist for this tensor in
        /// the tape.
        /// </summary>
        public UnorderedMap<long, long> tensor_usage_counts { get; set; }
        /// <summary>
        /// Maps from op ID to how many output tensors of this op still need to have
        /// their gradients computed.
        /// </summary>
        public UnorderedMap<long, long> op_missing_tensor { get; set; }

        public BackpropInitialState()
        {
            op_tape = new OpTape<BackwardFunction, TapeTensor>();
            tensor_usage_counts = new UnorderedMap<long, long>();
            op_missing_tensor = new UnorderedMap<long, long>();
        }
    }
}
