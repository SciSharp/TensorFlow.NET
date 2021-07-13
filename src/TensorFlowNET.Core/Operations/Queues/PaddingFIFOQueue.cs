/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using System.Linq;

namespace Tensorflow.Queues
{
    /// <summary>
    /// A FIFOQueue that supports batching variable-sized tensors by padding.
    /// </summary>
    public class PaddingFIFOQueue : QueueBase
    {
        public PaddingFIFOQueue(int capacity,
            TF_DataType[] dtypes,
            Shape[] shapes,
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
