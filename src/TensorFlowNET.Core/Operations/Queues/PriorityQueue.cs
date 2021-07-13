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

using System.Collections.Generic;
using System.Linq;
using static Tensorflow.Binding;

namespace Tensorflow.Queues
{
    public class PriorityQueue : QueueBase
    {
        public PriorityQueue(int capacity,
            TF_DataType[] dtypes,
            Shape[] shapes,
            string[] names = null,
            string shared_name = null,
            string name = "priority_queue")
        : base(dtypes: dtypes, shapes: shapes, names: names)
        {
            _queue_ref = gen_data_flow_ops.priority_queue_v2(
                component_types: dtypes,
                shapes: shapes,
                capacity: capacity,
                shared_name: shared_name,
                name: name);

            _name = _queue_ref.op.name.Split('/').Last();

            var dtypes1 = dtypes.ToList();
            dtypes1.Insert(0, TF_DataType.TF_INT64);
            _dtypes = dtypes1.ToArray();

            var shapes1 = shapes.ToList();
            shapes1.Insert(0, Shape.Null);
            _shapes = shapes1.ToArray();
        }

        public Operation enqueue_many<T>(long[] indexes, T[] vals, string name = null)
        {
            return tf_with(ops.name_scope(name, $"{_name}_EnqueueMany", vals), scope =>
            {
                var vals_tensor1 = _check_enqueue_dtypes(indexes);
                var vals_tensor2 = _check_enqueue_dtypes(vals);

                var tensors = new List<Tensor>();
                tensors.AddRange(vals_tensor1);
                tensors.AddRange(vals_tensor2);

                return gen_data_flow_ops.queue_enqueue_many_v2(_queue_ref, tensors.ToArray(), name: scope);
            });
        }

#pragma warning disable CS0108 // Member hides inherited member; missing new keyword
        public Tensor[] dequeue(string name = null)
#pragma warning restore CS0108 // Member hides inherited member; missing new keyword
        {
            Tensor[] ret;
            if (name == null)
                name = $"{_name}_Dequeue";

            if (_queue_ref.dtype == TF_DataType.TF_RESOURCE)
                ret = gen_data_flow_ops.queue_dequeue_v2(_queue_ref, _dtypes, name: name);
            else
                ret = gen_data_flow_ops.queue_dequeue(_queue_ref, _dtypes, name: name);

            return ret;
        }
    }
}
