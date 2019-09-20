using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Framework;
using static Tensorflow.Binding;

namespace Tensorflow.Queues
{
    /// <summary>
    /// A FIFOQueue that supports batching variable-sized tensors by padding.
    /// </summary>
    public class PaddingFIFOQueue : QueueBase
    {
        public PaddingFIFOQueue(int capacity, 
            TF_DataType[] dtypes, 
            TensorShape[] shapes, 
            string[] names = null,
            string shared_name = null,
            string name = "padding_fifo_queue")
            : base(dtypes: dtypes, shapes: shapes, names: names)
        {
            _queue_ref = gen_data_flow_ops.padding_fifo_queue_v2(
                component_types: dtypes,
                shapes: shapes,
                capacity: capacity,
                shared_name: shared_name,
                name: name);

            _name = _queue_ref.op.name.Split('/').Last();
        }
    }
}
