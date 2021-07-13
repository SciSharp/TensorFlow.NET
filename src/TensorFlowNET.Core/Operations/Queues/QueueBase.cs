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

using System;
using System.Collections.Generic;
using static Tensorflow.Binding;

namespace Tensorflow.Queues
{
    public class QueueBase
    {
        protected TF_DataType[] _dtypes;
        protected Shape[] _shapes;
        protected string[] _names;
        protected Tensor _queue_ref;
        protected string _name;

        public QueueBase(TF_DataType[] dtypes, Shape[] shapes, string[] names)
        {
            _dtypes = dtypes;
            _shapes = shapes;
            _names = names;
        }

        public Operation enqueue(Tensor val, string name = null)
        {
            return tf_with(ops.name_scope(name, $"{_name}_enqueue", val), scope =>
            {
                var vals = new[] { val };
                if (_queue_ref.dtype == TF_DataType.TF_RESOURCE)
                    return gen_data_flow_ops.queue_enqueue_v2(_queue_ref, vals, name: scope);
                else
                    return gen_data_flow_ops.queue_enqueue(_queue_ref, vals, name: scope);
            });
        }

        public Operation enqueue_many<T>(T[] vals, string name = null)
        {
            return tf_with(ops.name_scope(name, $"{_name}_EnqueueMany", vals), scope =>
            {
                var vals_tensor = _check_enqueue_dtypes(vals);
                return gen_data_flow_ops.queue_enqueue_many_v2(_queue_ref, vals_tensor, name: scope);
            });
        }

        protected Tensor[] _check_enqueue_dtypes(object vals)
        {
            var tensors = new List<Tensor>();

            switch (vals)
            {
                case int[][] vals1:
                    {
                        int i = 0;
                        foreach (var (val, dtype) in zip(vals1, _dtypes))
                            tensors.Add(ops.convert_to_tensor(val, dtype: dtype, name: $"component_{i++}"));
                    }
                    break;

                default:
                    var dtype1 = GetType().Name == "PriorityQueue" ? _dtypes[1] : _dtypes[0];
                    tensors.Add(ops.convert_to_tensor(vals, dtype: dtype1, name: $"component_0"));
                    break;
            }

            return tensors.ToArray();
        }

        /// <summary>
        /// Dequeues one element from this queue.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor dequeue(string name = null)
        {
            Tensor ret;
            if (name == null)
                name = $"{_name}_Dequeue";

            if (_queue_ref.dtype == TF_DataType.TF_RESOURCE)
                ret = gen_data_flow_ops.queue_dequeue_v2(_queue_ref, _dtypes, name: name)[0];
            else
                ret = gen_data_flow_ops.queue_dequeue(_queue_ref, _dtypes, name: name)[0];

            return ret;
        }

        public Tensor[] dequeue_many(int n, string name = null)
        {
            if (name == null)
                name = $"{_name}_DequeueMany";

            var ret = gen_data_flow_ops.queue_dequeue_many_v2(_queue_ref, n: n, component_types: _dtypes, name: name);
            //var op = ret[0].op;
            //var cv = tensor_util.constant_value(op.inputs[1]);
            //var batch_dim = new Dimension(cv);

            return _dequeue_return_value(ret);
        }

        public Tensor[] _dequeue_return_value(Tensor[] tensors)
        {
            if (_names != null)
                throw new NotImplementedException("");
            return tensors;
        }
    }
}
