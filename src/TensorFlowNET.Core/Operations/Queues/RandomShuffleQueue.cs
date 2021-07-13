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
    /// Create a queue that dequeues elements in a random order.
    /// </summary>
    public class RandomShuffleQueue : QueueBase
    {
        public RandomShuffleQueue(int capacity,
            int min_after_dequeue,
            TF_DataType[] dtypes,
            Shape[] shapes,
            string[] names = null,
            int? seed = null,
            string shared_name = null,
            string name = "random_shuffle_queue")
            : base(dtypes: dtypes, shapes: shapes, names: names)
        {
            var (seed1, seed2) = random_seed.get_seed(seed);
            if (!seed1.HasValue && !seed2.HasValue)
                (seed1, seed2) = (0, 0);


            _queue_ref = gen_data_flow_ops.random_shuffle_queue_v2(
                component_types: dtypes,
                shapes: shapes,
                capacity: capacity,
                min_after_dequeue: min_after_dequeue,
                seed: seed1.Value,
                seed2: seed2.Value,
                shared_name: shared_name,
                name: name);

            _name = _queue_ref.op.name.Split('/').Last();
        }
    }
}
