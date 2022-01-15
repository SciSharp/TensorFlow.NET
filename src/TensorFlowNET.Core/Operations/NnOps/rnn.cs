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

using Tensorflow.NumPy;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Framework;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace Tensorflow.Operations
{
    public class rnn
    {
        /// <summary>
        /// Creates a bidirectional recurrent neural network.
        /// </summary>
        public static (Tensor[], LSTMStateTuple, LSTMStateTuple) static_bidirectional_rnn(BasicLstmCell cell_fw,
            BasicLstmCell cell_bw,
            Tensor[] inputs,
            Tensor initial_state_fw = null,
            Tensor initial_state_bw = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            Tensor sequence_length = null,
            string scope = null)
        {
            if (inputs == null || inputs.Length == 0)
                throw new ValueError("inputs must not be empty");

            Tensor[] output_fw = null;
            Tensor[] output_bw = null;
            LSTMStateTuple output_state_fw = null;
            LSTMStateTuple output_state_bw = null;

            tf_with(tf.variable_scope(scope ?? "bidirectional_rnn"), delegate
            {
                // Forward direction
                tf_with(tf.variable_scope("fw"), fw_scope =>
                {
                    (output_fw, output_state_fw) = static_rnn(
                      cell_fw,
                      inputs,
                      initial_state_fw,
                      dtype,
                      sequence_length,
                      scope: fw_scope);
                });

                // backward direction
                tf_with(tf.variable_scope("bw"), bw_scope =>
                {
                    var reversed_inputs = _reverse_seq(inputs, sequence_length);
                    (output_bw, output_state_bw) = static_rnn(
                      cell_bw,
                      reversed_inputs,
                      initial_state_bw,
                      dtype,
                      sequence_length,
                      scope: bw_scope);
                });
            });

            output_bw = _reverse_seq(output_bw, sequence_length);

            var flat_outputs = zip(output_fw, output_bw)
                .Select(x => array_ops.concat(new[] { x.Item1, x.Item2 }, 1))
                .ToArray();

            return (flat_outputs, output_state_fw, output_state_bw);
        }

        private static Tensor[] _reverse_seq(Tensor[] input_seq, Tensor lengths)
        {
            if (lengths == null)
                return input_seq.Reverse().ToArray();

            throw new NotImplementedException("_reverse_seq");
        }

        public static (Tensor[], LSTMStateTuple) static_rnn(BasicLstmCell cell,
            Tensor[] inputs,
            Tensor initial_state,
            TF_DataType dtype = TF_DataType.DtInvalid,
            Tensor sequence_length = null,
            VariableScope scope = null)
        {
            List<Tensor> outputs = new List<Tensor>();
            object state = null;

            // Create a new scope in which the caching device is either
            // determined by the parent scope, or is set to place the cached
            // Variable using the same placement as for the rest of the RNN.
            if (scope == null)
                tf_with(tf.variable_scope("rnn"), varscope =>
                {
                    throw new NotImplementedException("static_rnn");
                });
            else
                tf_with(tf.variable_scope(scope), scope1 =>
                {
                    Dimension fixed_batch_size = null;
                    Dimension batch_size = null;
                    Tensor batch_size_tensor = null;
                    VariableScope varscope = scope1;
                    // Obtain the first sequence of the input
                    var first_input = inputs[0];
                    if (first_input.shape.ndim != 1)
                    {
                        var input_shape = first_input.shape.with_rank_at_least(2);
                        fixed_batch_size = input_shape.dims[0];
                        var flat_inputs = nest.flatten2(inputs);
                        foreach (var flat_input in flat_inputs)
                        {
                            input_shape = flat_input.shape.with_rank_at_least(2);
                            batch_size = tensor_shape.dimension_at_index(input_shape, 0);
                            var input_size = input_shape[new Slice(1)];
                            fixed_batch_size.merge_with(batch_size);
                            foreach (var (i, size) in enumerate(input_size.dims))
                            {
                                if (size < 0)
                                    throw new ValueError($"Input size (dimension {i} of inputs) must be accessible via " +
                                        "shape inference, but saw value None.");
                            }
                        }
                    }
                    else
                        fixed_batch_size = first_input.shape.with_rank_at_least(1).dims[0];

                    if (tensor_shape.dimension_value(fixed_batch_size) >= 0)
                        batch_size = tensor_shape.dimension_value(fixed_batch_size);
                    else
                        batch_size_tensor = array_ops.shape(first_input)[0];

                    if (initial_state != null)
                        state = initial_state;
                    else
                    {
                        state = cell.get_initial_state(batch_size: batch_size_tensor, dtype: dtype);
                    }

                    Tensor output = null;
                    if (state is LSTMStateTuple state_tuple)
                    {
                        foreach (var (time, input_) in enumerate(inputs))
                        {
                            if (time > 0)
                                varscope.reuse_variables();
                            if (sequence_length != null)
                                throw new NotImplementedException("static_rnn");

                            var results = cell.__call__(input_, state_tuple);
                            (output, state_tuple) = (results[1], new LSTMStateTuple(results[0], results[1]));
                            outputs.Add(output);
                        }
                    }
                });

            return (outputs.ToArray(), state as LSTMStateTuple);
        }

        public static (Tensor, Tensor) dynamic_rnn(RnnCell cell, Tensor inputs_tensor,
            Tensor sequence_length = null, Tensor initial_state = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            int? parallel_iterations = null, bool swap_memory = false, bool time_major = false)
        {
            return tf_with(tf.variable_scope("rnn"), scope =>
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
                    state = cell.get_initial_state(batch_size: batch_size, dtype: dtype) as Tensor;

                var inputs = nest.pack_sequence_as(structure: inputs_tensor, flat_sequence: flat_input);

                var (outputs, final_state) = _dynamic_rnn_loop(
                    cell,
                    inputs as Tensor,
                    state,
                    parallel_iterations: parallel_iterations.Value,
                    swap_memory: swap_memory,
                    sequence_length: sequence_length,
                    dtype: dtype);

                if (!time_major)
                    outputs = nest.map_structure(_transpose_batch_time, outputs);

                return (outputs, final_state);
            });
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
        private static (Tensor, Tensor) _dynamic_rnn_loop(RnnCell cell, Tensor inputs, Tensor initial_state,
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
            var inputs_got_shape = flat_input.Select(input_ => input_.shape.with_rank_at_least(3)).ToArray();

            var dims = inputs_got_shape[0].dims.Take(2).ToArray();
            var (const_time_steps, const_batch_size) = (dims[0], dims[1]);

            foreach (var shape in inputs_got_shape)
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

            Func<string, Shape, TF_DataType, TensorArray> _create_ta = (name, element_shape, dtype_) =>
            {
                var ta = tf.TensorArray(dtype: dtype_,
                                        size: time_steps,
                                        element_shape: element_shape);
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
                        new Shape(const_batch_size).concatenate(
                            _maybe_tensor_shape_from_tensor(out_size)),
                        _infer_state_dtype(dtype, state)));
                }

                foreach (var (i, flat_input_i) in enumerate(flat_input))
                {
                    input_ta.Add(_create_ta($"input_{i}",
                        new Shape(flat_input_i.dims.Skip(1).ToArray()),
                        flat_input_i.dtype));
                }

                input_ta = zip(input_ta, flat_input).Select(x =>
                {
                    var (ta, input_) = (x.Item1, x.Item2);
                    return ta.unstack(input_);
                }).ToList();
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
                Tensor[] input_t = null;
                var (time1, output_ta_t, state1) = (item.time, item.output_ta_t, item.state);
                if (in_graph_mode)
                {
                    input_t = input_ta.Select(ta => ta.read(time1)).ToArray();
                    // Restore some shape information
                    foreach (var (input_, shape) in zip(input_t, inputs_got_shape))
                        input_.shape = shape[new Slice(1)];
                }
                else
                {
                    // input_t = tuple(ta[time.numpy()] for ta in input_ta)
                }

                var input_t_t = nest.pack_sequence_as2(structure: inputs, flat_sequence: input_t);
                // Keras RNN cells only accept state as list, even if it's a single tensor.
                // var is_keras_rnn_cell = _is_keras_rnn_cell(cell);
                Tensor[] outputs = null;
                if (sequence_length != null)
                    throw new NotImplementedException("sequence_length != null");
                /*else
                    outputs = cell.__call__(input_t_t, state: state1);*/

                var (output, new_state) = (outputs[0], outputs[1]);
                // Keras cells always wrap state as list, even if it's a single tensor.
                // if(is_keras_rnn_cell && len(new_state)) == 1
                // Pack state if using state tuples
                outputs = nest.flatten2(output).Select(x => x as Tensor).ToArray();

                output_ta_t = zip(output_ta_t, outputs).Select(x =>
                {
                    var (ta, @out) = (x.Item1, x.Item2);
                    return ta.write(item.time, @out);
                }).ToArray();

                return new BodyItemInRnnWhileLoop(item.time + 1, output_ta_t, new_state);
            };

            var while_loop_result = control_flow_ops.while_loop(
              cond: cond,
              body: _time_step,
              loop_vars: new BodyItemInRnnWhileLoop(time, output_ta.ToArray(), state),
              parallel_iterations: parallel_iterations,
              maximum_iterations: time_steps,
              swap_memory: swap_memory);

            (_, TensorArray[] output_final_ta, Tensor final_state) = (while_loop_result.time, while_loop_result.output_ta_t, while_loop_result.state);

            // Unpack final output if not using output tuples.
            var final_outputs = output_final_ta.Select(ta => ta.stack()).ToArray();
            // Restore some shape information
            foreach (var (output, output_size) in zip(final_outputs, flat_output_size))
            {
                var shape = rnn_cell_impl._concat(new int[] { (int)const_time_steps, (int)const_batch_size }, output_size, @static: true);
                output.shape = shape;
            }

            return (final_outputs[0], final_state);
        }

        private static Shape _maybe_tensor_shape_from_tensor(Tensor shape)
            => shape.shape;

        private static Shape _maybe_tensor_shape_from_tensor(int shape)
            => new Shape(shape);

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
            var x_static_shape = x.shape;
            if (x_static_shape.ndim == 1)
                return x;

            var x_rank = array_ops.rank(x);
            var con1 = new object[]
            {
                new []{1, 0 },
                math_ops.range(2, x_rank)
            };
            var x_t = array_ops.transpose(x, array_ops.concat(con1, 0));

            var dims = new long[] { x_static_shape.dims[1], x_static_shape.dims[0] }
                .ToList();
            dims.AddRange(x_static_shape.dims.Skip(2));
            var shape = new Shape(dims.ToArray());

            x_t.shape = shape;

            return x_t;
        }

        /// <summary>
        /// Get static input batch size if available, with fallback to the dynamic one.
        /// </summary>
        /// <param name="flat_input"></param>
        /// <returns></returns>
        private static Tensor _best_effort_input_batch_size(List<Tensor> flat_input)
        {
            foreach (var input_ in flat_input)
            {
                var shape = input_.shape;
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
