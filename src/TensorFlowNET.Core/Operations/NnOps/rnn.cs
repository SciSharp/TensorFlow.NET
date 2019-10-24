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
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace Tensorflow.Operations
{
    internal class rnn
    {
        public static (Tensor, Tensor) dynamic_rnn(RNNCell cell, Tensor inputs_tensor,
            Tensor sequence_length = null, Tensor initial_state = null, 
            TF_DataType dtype = TF_DataType.DtInvalid,
            int? parallel_iterations = null, bool swap_memory = false, bool time_major = false)
        {
            tf_with(tf.variable_scope("rnn"), scope =>
            {
                VariableScope varscope = scope;
                var flat_input = nest.flatten(inputs_tensor);

                if (!time_major)
                {
                    flat_input = flat_input.Select(x => ops.convert_to_tensor(x)).ToList();
                    flat_input = flat_input.Select(x => _transpose_batch_time(x)).ToList();
                }

                parallel_iterations = parallel_iterations ?? 32;

                if (sequence_length != null)
                    throw new NotImplementedException("dynamic_rnn sequence_length has value");

                var batch_size = _best_effort_input_batch_size(flat_input);

                Tensor state = null;
                if (initial_state != null)
                    state = initial_state;
                else
                    state = cell.get_initial_state(batch_size: batch_size, dtype: dtype);

                var inputs = nest.pack_sequence_as(structure: inputs_tensor, flat_sequence: flat_input);

                var (outputs, final_state) = _dynamic_rnn_loop(
                    cell,
                    inputs as Tensor,
                    state,
                    parallel_iterations: parallel_iterations.Value,
                    swap_memory: swap_memory,
                    sequence_length: sequence_length,
                    dtype: dtype);
            });

            throw new NotImplementedException("");
        }

        /// <summary>
        /// Internal implementation of Dynamic RNN.
        /// </summary>
        /// <param name="cell"></param>
        /// <param name="inputs"></param>
        /// <param name="initial_state"></param>
        /// <param name="parallel_iterations"></param>
        /// <param name="swap_memory"></param>
        /// <param name="sequence_length"></param>
        /// <param name="dtype"></param>
        /// <returns></returns>
        private static (Tensor, Tensor) _dynamic_rnn_loop(RNNCell cell, Tensor inputs, Tensor initial_state,
            int parallel_iterations, bool swap_memory, Tensor sequence_length = null, TF_DataType dtype = TF_DataType.DtInvalid)
        {
            var state = initial_state;
            var state_size = cell.state_size;

            var flat_input = nest.flatten(inputs);
            var flat_output_size = nest.flatten(cell.output_size);

            // Construct an initial output
            var input_shape = array_ops.shape(flat_input[0]);
            var time_steps = input_shape.slice(0);
            var batch_size = _best_effort_input_batch_size(flat_input);
            var inputs_got_shape = flat_input.Select(input_ => input_.TensorShape.with_rank_at_least(3)).ToArray();

            var dims = inputs_got_shape[0].dims.Take(2).ToArray();
            var (const_time_steps, const_batch_size) = (dims[0], dims[1]);

            foreach(var shape in inputs_got_shape)
            {
                if (shape.dims[2] == -1)
                    throw new ValueError("Input size (depth of inputs) must be accessible via shape inference," +
                        " but saw value None.");

                var got_time_steps = shape.dims[0];
                var got_batch_size = shape.dims[1];

                if (const_time_steps != got_time_steps)
                    throw new ValueError("Time steps is not the same for all the elements in the input in a " +
                        "batch.");

                if (const_batch_size != got_batch_size)
                    throw new ValueError("Batch_size is not the same for all the elements in the input.");
            }

            Func<int, Tensor> _create_zero_arrays = (size_) =>
            {
                var size = rnn_cell_impl._concat(batch_size, size_);
                return array_ops.zeros(
                    array_ops.stack(size), dtype: _infer_state_dtype(dtype, state));
            };

            // Prepare dynamic conditional copying of state & output
            var flat_zero_output = flat_output_size.Select(output => _create_zero_arrays(output)).ToArray();
            var zero_output = nest.pack_sequence_as(structure: cell.output_size, flat_sequence: flat_zero_output);

            Tensor min_sequence_length = null, max_sequence_length = null;
            if (sequence_length != null)
            {
                min_sequence_length = math_ops.reduce_min(sequence_length);
                max_sequence_length = math_ops.reduce_max(sequence_length);
            }
            else
            {
                max_sequence_length = time_steps;
            }

            var time = array_ops.constant(0, dtype: dtypes.int32, name: "time");

            string base_name = null;
            tf_with(ops.name_scope("dynamic_rnn"), scope => base_name = scope);

            Func<string, TensorShape, TF_DataType, TensorArray> _create_ta = (name, element_shape, dtype_) =>
            {
                var ta = new TensorArray(dtype: dtype_,
                                        size: time_steps,
                                        element_shape: element_shape,
                                        tensor_array_name: base_name + name);
                return ta;
            };

            bool in_graph_mode = true;
            var output_ta = new List<TensorArray>();
            var input_ta = new List<TensorArray>();
            if (in_graph_mode)
            {
                foreach (var (i, out_size) in enumerate(flat_output_size))
                {
                    output_ta.Add(_create_ta($"output_{i}",
                        new TensorShape(const_batch_size).concatenate(
                            _maybe_tensor_shape_from_tensor(out_size)),
                        _infer_state_dtype(dtype, state)));
                }

                foreach (var (i, flat_input_i) in enumerate(flat_input))
                {
                    input_ta.Add(_create_ta($"input_{i}",
                        new TensorShape(flat_input_i.dims.Skip(1).ToArray()),
                        flat_input_i.dtype));
                }

                for (int i = 0; i < input_ta.Count; i++)
                {
                    var (ta, input_) = (input_ta[i], flat_input[i]);
                    ta.unstack(input_);
                }
            }

            // Make sure that we run at least 1 step, if necessary, to ensure
            // the TensorArrays pick up the dynamic shape.
            Tensor loop_bound = null;
            if (in_graph_mode)
                loop_bound = math_ops.minimum(
                    time_steps, math_ops.maximum(1, max_sequence_length));

            Func<BodyItemInRnnWhileLoop, Tensor> cond = (item) =>
            {
                return item.time < loop_bound;
            };

            // Take a time step of the dynamic RNN.
            Func<BodyItemInRnnWhileLoop, BodyItemInRnnWhileLoop> _time_step = (item) =>
            {
                throw new NotImplementedException("");
            };

            control_flow_ops.while_loop(
              cond: cond,
              body: _time_step,
              loop_vars: new BodyItemInRnnWhileLoop(time, output_ta.ToArray(), state),
              parallel_iterations: parallel_iterations,
              maximum_iterations: time_steps,
              swap_memory: swap_memory);

            throw new NotImplementedException("");
        }

        private static TensorShape _maybe_tensor_shape_from_tensor(Tensor shape)
            => shape.TensorShape;

        private static TensorShape _maybe_tensor_shape_from_tensor(int shape)
            => new TensorShape(shape);

        private static TF_DataType _infer_state_dtype(TF_DataType explicit_dtype, Tensor state)
        {
            if (explicit_dtype != TF_DataType.DtInvalid)
                return explicit_dtype;

            throw new NotImplementedException("_infer_state_dtype");
        }

        /// <summary>
        /// Transposes the batch and time dimensions of a Tensor.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static Tensor _transpose_batch_time(Tensor x)
        {
            var x_static_shape = x.TensorShape;
            if (x_static_shape.ndim == 1)
                return x;

            var x_rank = array_ops.rank(x);
            var con1 = new object[]
            {
                new []{1, 0 },
                math_ops.range(2, x_rank)
            };
            var x_t = array_ops.transpose(x, array_ops.concat(con1, 0));

            var dims = new int[] { x_static_shape.dims[1], x_static_shape.dims[0] }
                .ToList();
            dims.AddRange(x_static_shape.dims.Skip(2));
            var shape = new TensorShape(dims.ToArray());

            x_t.set_shape(shape);

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
                if (shape.ndim < 0)
                    continue;
                if (shape.ndim < 2)
                    throw new ValueError($"Expected input tensor {input_.name} to have rank at least 2");

                var batch_size = shape.dims[1];
                if (batch_size > -1)
                    throw new ValueError("_best_effort_input_batch_size batch_size > -1");
                    //return batch_size;
            }

            return array_ops.shape(flat_input[0]).slice(1);
        }
    }
}
