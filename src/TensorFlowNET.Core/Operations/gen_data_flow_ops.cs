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

namespace Tensorflow
{
    public class gen_data_flow_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        public static Tensor dynamic_stitch(Tensor[] indices, Tensor[] data, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("DynamicStitch", name, new { indices, data });

            return _op.output;
        }

        public static (Tensor, Tensor) tensor_array_v3(Tensor size, TF_DataType dtype = TF_DataType.DtInvalid, 
            int[] element_shape = null, bool dynamic_size = false, bool clear_after_read = true, 
            bool identical_element_shapes = false, string tensor_array_name = "tensor_array_name", string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("TensorArrayV3", name, new
            {
                size,
                dtype,
                element_shape,
                dynamic_size,
                clear_after_read,
                identical_element_shapes,
                tensor_array_name
            });

            return (null, null);
        }

        public static Tensor padding_fifo_queue_v2(TF_DataType[] component_types, TensorShape[] shapes, 
            int capacity = -1, string container = "", string shared_name = "", 
            string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("PaddingFIFOQueueV2", name, new
            {
                component_types,
                shapes,
                capacity,
                container,
                shared_name
            });

            return _op.output;
        }

        public static Tensor fifo_queue_v2(TF_DataType[] component_types, TensorShape[] shapes,
            int capacity = -1, string container = "", string shared_name = "",
            string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("FIFOQueueV2", name, new
            {
                component_types,
                shapes,
                capacity,
                container,
                shared_name
            });

            return _op.output;
        }

        public static Tensor priority_queue_v2(TF_DataType[] component_types, TensorShape[] shapes,
            int capacity = -1, string container = "", string shared_name = "",
            string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("PriorityQueueV2", name, new
            {
                component_types,
                shapes,
                capacity,
                container,
                shared_name
            });

            return _op.output;
        }

        public static Tensor random_shuffle_queue_v2(TF_DataType[] component_types, TensorShape[] shapes,
            int capacity = -1, int min_after_dequeue = 0, int seed = 0, int seed2 = 0,
            string container = "", string shared_name = "", string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("RandomShuffleQueueV2", name, new
            {
                component_types,
                shapes,
                capacity,
                min_after_dequeue,
                seed,
                seed2,
                container,
                shared_name
            });

            return _op.output;
        }

        public static Operation queue_enqueue(Tensor handle, Tensor[] components, int timeout_ms = -1, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("QueueEnqueue", name, new
            {
                handle,
                components,
                timeout_ms
            });

            return _op;
        }

        public static Operation queue_enqueue_v2(Tensor handle, Tensor[] components, int timeout_ms = -1, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("QueueEnqueueV2", name, new
            {
                handle,
                components,
                timeout_ms
            });

            return _op;
        }

        public static Tensor[] queue_dequeue_v2(Tensor handle, TF_DataType[] component_types, int timeout_ms = -1, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("QueueDequeueV2", name, new
            {
                handle,
                component_types,
                timeout_ms
            });

            return _op.outputs;
        }

        public static Tensor[] queue_dequeue(Tensor handle, TF_DataType[] component_types, int timeout_ms = -1, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("QueueDequeue", name, new
            {
                handle,
                component_types,
                timeout_ms
            });

            return _op.outputs;
        }

        public static Operation queue_enqueue_many_v2(Tensor handle, Tensor[] components, int timeout_ms = -1, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("QueueEnqueueManyV2", name, new
            {
                handle,
                components,
                timeout_ms
            });

            return _op;
        }

        public static Tensor[] queue_dequeue_many_v2(Tensor handle, int n, TF_DataType[] component_types, int timeout_ms = -1, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("QueueDequeueManyV2", name, new
            {
                handle,
                n,
                component_types,
                timeout_ms
            });

            return _op.outputs;
        }
    }
}
