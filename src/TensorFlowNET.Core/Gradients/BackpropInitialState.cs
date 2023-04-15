using Tensorflow.Util;

namespace Tensorflow.Gradients
{
    public class BackpropInitialState
    {
        public OpTape op_tape { get; set; }
        /// <summary>
        /// Map from tensor to how many references still exist for this tensor in
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
            op_tape = new OpTape();
            tensor_usage_counts = new UnorderedMap<long, long>();
            op_missing_tensor = new UnorderedMap<long, long>();
        }
    }
}
