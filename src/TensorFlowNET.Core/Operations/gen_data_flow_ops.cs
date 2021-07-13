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

using static Tensorflow.Binding;

namespace Tensorflow
{
    public class gen_data_flow_ops
    {
        public static Tensor dynamic_stitch(Tensor[] indices, Tensor[] data, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("DynamicStitch", name, new { indices, data });

            return _op.output;
        }

        public static Tensor[] dynamic_partition(Tensor data, Tensor partitions, int num_partitions,
            string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("DynamicPartition", name, new
            {
                data,
                partitions,
                num_partitions
            });

            return _op.outputs;
        }

        public static (Tensor, Tensor) tensor_array_v3<T>(T size, TF_DataType dtype = TF_DataType.DtInvalid,
            Shape element_shape = null, bool dynamic_size = false, bool clear_after_read = true,
            bool identical_element_shapes = false, string tensor_array_name = "", string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("TensorArrayV3", name, new
            {
                size,
                dtype,
                element_shape,
                dynamic_size,
                clear_after_read,
                identical_element_shapes,
                tensor_array_name
            });

            return (_op.outputs[0], _op.outputs[1]);
        }

        public static Tensor tensor_array_scatter_v3(Tensor handle, Tensor indices, Tensor value,
            Tensor flow_in, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("TensorArrayScatterV3", name, new
            {
                handle,
                indices,
                value,
                flow_in
            });

            return _op.output;
        }

        public static Tensor padding_fifo_queue_v2(TF_DataType[] component_types, Shape[] shapes,
            int capacity = -1, string container = "", string shared_name = "",
            string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("PaddingFIFOQueueV2", name, new
            {
                component_types,
                shapes,
                capacity,
                container,
                shared_name
            });

            return _op.output;
        }

        public static Tensor fifo_queue_v2(TF_DataType[] component_types, Shape[] shapes,
            int capacity = -1, string container = "", string shared_name = "",
            string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("FIFOQueueV2", name, new
            {
                component_types,
                shapes,
                capacity,
                container,
                shared_name
            });

            return _op.output;
        }

        public static Tensor priority_queue_v2(TF_DataType[] component_types, Shape[] shapes,
            int capacity = -1, string container = "", string shared_name = "",
            string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("PriorityQueueV2", name, new
            {
                component_types,
                shapes,
                capacity,
                container,
                shared_name
            });

            return _op.output;
        }

        public static Tensor random_shuffle_queue_v2(TF_DataType[] component_types, Shape[] shapes,
            int capacity = -1, int min_after_dequeue = 0, int seed = 0, int seed2 = 0,
            string container = "", string shared_name = "", string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("RandomShuffleQueueV2", name, new
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
            var _op = tf.OpDefLib._apply_op_helper("QueueEnqueue", name, new
            {
                handle,
                components,
                timeout_ms
            });

            return _op;
        }

        public static Operation queue_enqueue_v2(Tensor handle, Tensor[] components, int timeout_ms = -1, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("QueueEnqueueV2", name, new
            {
                handle,
                components,
                timeout_ms
            });

            return _op;
        }

        public static Tensor[] queue_dequeue_v2(Tensor handle, TF_DataType[] component_types, int timeout_ms = -1, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("QueueDequeueV2", name, new
            {
                handle,
                component_types,
                timeout_ms
            });

            return _op.outputs;
        }

        public static Tensor[] queue_dequeue(Tensor handle, TF_DataType[] component_types, int timeout_ms = -1, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("QueueDequeue", name, new
            {
                handle,
                component_types,
                timeout_ms
            });

            return _op.outputs;
        }

        public static Operation queue_enqueue_many_v2(Tensor handle, Tensor[] components, int timeout_ms = -1, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("QueueEnqueueManyV2", name, new
            {
                handle,
                components,
                timeout_ms
            });

            return _op;
        }

        public static Tensor[] queue_dequeue_many_v2(Tensor handle, int n, TF_DataType[] component_types, int timeout_ms = -1, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("QueueDequeueManyV2", name, new
            {
                handle,
                n,
                component_types,
                timeout_ms
            });

            return _op.outputs;
        }

        /// <summary>
        /// Read an element from the TensorArray into output `value`.
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="index"></param>
        /// <param name="flow_in"></param>
        /// <param name="dtype"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor tensor_array_read_v3(Tensor handle, Tensor index, Tensor flow_in, TF_DataType dtype, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("TensorArrayReadV3", name, new
            {
                handle,
                index,
                flow_in,
                dtype
            });

            return _op.output;
        }

        public static Tensor tensor_array_write_v3(Tensor handle, Tensor index, Tensor value, Tensor flow_in, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("TensorArrayWriteV3", name, new
            {
                handle,
                index,
                value,
                flow_in
            });

            return _op.output;
        }

        public static Tensor tensor_array_size_v3(Tensor handle, Tensor flow_in, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("TensorArraySizeV3", name, new
            {
                handle,
                flow_in
            });

            return _op.output;
        }

        public static Tensor tensor_array_gather_v3(Tensor handle, Tensor indices, Tensor flow_in,
            TF_DataType dtype, Shape element_shape = null, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("TensorArrayGatherV3", name, new
            {
                handle,
                indices,
                dtype,
                element_shape,
                flow_in
            });

            return _op.output;
        }

        public static Tensor stack_v2(Tensor max_size, TF_DataType elem_type, string stack_name = "",
            string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("StackV2", name, new
            {
                max_size,
                elem_type,
                stack_name
            });

            return _op.output;
        }

        public static Tensor stack_push_v2(Tensor handle, Tensor elem, bool swap_memory = false,
            string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("StackPushV2", name, new
            {
                handle,
                elem,
                swap_memory
            });

            return _op.output;
        }

        public static Tensor stack_pop_v2(Tensor handle, TF_DataType elem_type, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("StackPopV2", name, new
            {
                handle,
                elem_type
            });

            return _op.output;
        }
    }
}
