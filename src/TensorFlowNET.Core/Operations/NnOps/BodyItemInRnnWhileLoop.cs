using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Operations
{
    internal class BodyItemInRnnWhileLoop
    {
        /// <summary>
        /// int32 scalar Tensor.
        /// </summary>
        public Tensor time { get; set; }
        /// <summary>
        /// List of `TensorArray`s that represent the output.
        /// </summary>
        public TensorArray[] output_ta_t { get; set; }
        /// <summary>
        /// nested tuple of vector tensors that represent the state.
        /// </summary>
        public Tensor state { get; set; }

        public BodyItemInRnnWhileLoop(Tensor time, TensorArray[] output_ta_t, Tensor state)
        {
            this.time = time;
            this.output_ta_t = output_ta_t;
            this.state = state;
        }

        public static implicit operator (Tensor, TensorArray[], Tensor)(BodyItemInRnnWhileLoop item)
            => (item.time, item.output_ta_t, item.state);
    }
}
