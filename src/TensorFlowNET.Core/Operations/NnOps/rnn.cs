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
using System.Linq;
using static Tensorflow.Python;
using Tensorflow.Util;

namespace Tensorflow.Operations
{
    internal class rnn
    {
        public static (Tensor, Tensor) dynamic_rnn(RNNCell cell, Tensor inputs,
            int? sequence_length = null, Tensor initial_state = null, 
            TF_DataType dtype = TF_DataType.DtInvalid,
            int? parallel_iterations = null, bool swap_memory = false, bool time_major = false)
        {
            with(tf.variable_scope("rnn"), scope =>
            {
                VariableScope varscope = scope;
                var flat_input = nest.flatten(inputs);

                if (!time_major)
                {
                    flat_input = flat_input.Select(x => ops.convert_to_tensor(x)).ToList();
                    flat_input = flat_input.Select(x => _transpose_batch_time(x)).ToList();
                }

                parallel_iterations = parallel_iterations ?? 32;

                if (sequence_length.HasValue)
                    throw new NotImplementedException("dynamic_rnn sequence_length has value");

                var batch_size = _best_effort_input_batch_size(flat_input);

                if (initial_state != null)
                {
                    var state = initial_state;
                }
                else
                {
                    cell.get_initial_state(batch_size: batch_size, dtype: dtype);
                }
            });

            throw new NotImplementedException("");
        }

        /// <summary>
        /// Transposes the batch and time dimensions of a Tensor.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static Tensor _transpose_batch_time(Tensor x)
        {
            var x_static_shape = x.TensorShape;
            if (x_static_shape.NDim == 1)
                return x;

            var x_rank = array_ops.rank(x);
            var con1 = new object[]
            {
                new []{1, 0 },
                math_ops.range(2, x_rank)
            };
            var x_t = array_ops.transpose(x, array_ops.concat(con1, 0));

            var dims = new int[] { x_static_shape.Dimensions[1], x_static_shape.Dimensions[0] }
                .ToList();
            dims.AddRange(x_static_shape.Dimensions.Skip(2));
            var shape = new TensorShape(dims.ToArray());

            x_t.SetShape(shape);

            return x_t;
        }

        /// <summary>
        /// Get static input batch size if available, with fallback to the dynamic one.
        /// </summary>
        /// <param name="flat_input"></param>
        /// <returns></returns>
        private static Tensor _best_effort_input_batch_size(List<Tensor> flat_input)
        {
            foreach(var input_ in flat_input)
            {
                var shape = input_.TensorShape;
                if (shape.NDim < 0)
                    continue;
                if (shape.NDim < 2)
                    throw new ValueError($"Expected input tensor {input_.name} to have rank at least 2");

                var batch_size = shape.Dimensions[1];
                if (batch_size > -1)
                    throw new ValueError("_best_effort_input_batch_size batch_size > -1");
                    //return batch_size;
            }

            return array_ops.shape(flat_input[0]).slice(1);
        }
    }
}
