using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Operations
{
    internal class BodyItemInRnnWhileLoop : ICanBeFlattened, IPackable
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

        public object[] Flatten()
        {
            var elements = new List<object> { time };
            elements.AddRange(output_ta_t);
            elements.Add(state);
            return elements.ToArray();
        }

        public void Pack(object[] sequences)
        {
            time = sequences[0] as Tensor;
            output_ta_t = new[] { sequences[1] as TensorArray };
            state = sequences[2] as Tensor;
        }
    }
}
