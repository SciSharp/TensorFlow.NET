using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow.Queues
{
    public class RandomShuffleQueue : QueueBase
    {
        public RandomShuffleQueue(int capacity,
            TF_DataType[] dtypes,
            TensorShape[] shapes,
            string[] names = null,
            string shared_name = null,
            string name = "randomshuffle_fifo_queue")
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
