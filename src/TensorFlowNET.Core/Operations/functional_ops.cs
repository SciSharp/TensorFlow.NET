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
using Tensorflow.Framework;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class functional_ops
    {
        public static Tensor scan(
            Func<Tensor, Tensor, Tensor> fn,
            Tensor elems,
            Tensor initializer = null,
            int parallel_iterations = 10,
            bool back_prop = true,
            bool swap_memory = false,
            bool infer_shape = true,
            bool reverse = false,
            string name = null)
        {
            bool input_is_sequence = nest.is_sequence(elems);

            List<Tensor> input_flatten(Tensor x) => input_is_sequence ? nest.flatten(x) : new List<Tensor> {x};
            Tensor input_pack(List<Tensor> x) => input_is_sequence ? (Tensor)nest.pack_sequence_as(elems, x) : x[0];

            bool output_is_sequence;
            Func<Tensor, List<Tensor>> output_flatten;
            Func<List<Tensor>, Tensor> output_pack;
            if (initializer == null)
            {
                output_is_sequence = input_is_sequence;
                output_flatten = input_flatten;
                output_pack = input_pack;
            }
            else
            {
                output_is_sequence = nest.is_sequence(initializer);
                output_flatten = (x) => output_is_sequence ? nest.flatten(x) : new List<Tensor> {x};
                output_pack = (x) => output_is_sequence ? (Tensor)nest.pack_sequence_as(initializer, x) : x[0];
            }

            var elems_flat = input_flatten(elems);

            bool in_graph_mode = true; // todo !context.executing_eagerly()

            return tf_with(ops.name_scope(name, "scan", new { elems_flat }), scope =>
            {
                // todo tf.net doesn't expose .caching_device
                //if (in_graph_mode)
                //{
                //    // Any get_variable calls in fn will cache the first call locally
                //    // and not issue repeated network I/O requests for each iteration.
                //    var varscope = variable_scope.get_variable_scope();
                //    bool varscope_caching_device_was_none = false;
                //    if (varscope.caching_device = null)
                //    {
                //        //      varscope.set_caching_device(lambda op: op.device)
                //        //      varscope_caching_device_was_none = True
                //    }
                //}

                elems_flat = elems_flat.Select(elem => ops.convert_to_tensor(elem, name: "elem")).ToList();

                var n = tensor_shape.dimension_value(elems_flat[0].shape[0]);

                // todo python had the below but dimension_value returns int which can't be null
                //if (n == null)
                //{
                //    n = array_ops.shape(elems_flat[0])[0];
                //}

                var elems_ta = elems_flat.Select(elem => new TensorArray(
                    elem.dtype,
                    size: tf.constant(n),
                    dynamic_size: false,
                    element_shape: elem.shape.Skip(1).ToArray(),
                    infer_shape: true)).ToList();

                for (int index = 0; index < elems_ta.Count; index++)
                {
                    elems_ta[index].unstack(elems_flat[index]);
                }

                List<Tensor> a_flat;
                int i;
                if (initializer == null)
                {
                    a_flat = elems_ta.Select(elem => elem.read(tf.constant(reverse ? n - 1 : 0))).ToList();
                    i = 1;
                }
                else
                {
                    List<Tensor> initializer_flat = output_flatten(initializer);
                    a_flat = initializer_flat.Select(init => ops.convert_to_tensor(init)).ToList();
                    i = 0;
                }

                var accs_ta = a_flat.Select(init => new TensorArray(
                    dtype: init.dtype,
                    size: tf.constant(n),
                    element_shape: infer_shape ? init.shape : null,
                    dynamic_size: false,
                    infer_shape: infer_shape)).ToList();

                if (initializer == null)
                {
                    for (int index = 0; index < accs_ta.Count; index++)
                    {
                        accs_ta[index].write(tf.constant(reverse ? n - 1 : 0), a_flat[index]);
                    }
                }

                (int, List<Tensor>, List<TensorArray>) compute(ValueTuple<int, List<Tensor>, List<TensorArray>> tuple)
                {

                    (int _i, List<Tensor> a_flat_, List<TensorArray> tas) = tuple;

                    var packed_elems = input_pack(elems_ta.Select(elem_ta => elem_ta.read(tf.constant(_i))).ToList());
                    var packed_a = output_pack(a_flat_);
                    var a_out = fn((Tensor)packed_a, (Tensor)packed_elems); // todo brendan are these casts legal?

                    var flat_a_out = output_flatten(a_out);
                    for (int j = 0; j < tas.Count; j++)
                    {
                        tas[j].write(tf.constant(i), flat_a_out[j]);
                    }

                    var next_i = reverse ? _i-- : _i++;
                    return (next_i, flat_a_out, tas);
                }

                int initial_i;
                Func<(int, List<Tensor>, List<TensorArray>), Tensor> condition;
                if (reverse)
                {
                    initial_i = n - 1 - i;
                    condition = x => tf.constant(x.Item1 >= 0);
                }
                else
                {
                    initial_i = i;
                    condition = x => tf.constant(x.Item1 < n);
                }

                (_, _, List<TensorArray> r_a) =
                    control_flow_ops.while_loop(
                        condition,
                        compute,
                        (initial_i, a_flat, accs_ta),
                        parallel_iterations: parallel_iterations,
                        back_prop: back_prop,
                        swap_memory: swap_memory,
                        maximum_iterations: tf.constant(n));

                var results_flat = r_a.Select(r => r.stack()).ToList();

                var n_static = new Dimension(tensor_shape.dimension_value(elems_flat[0].shape[0]));
                
                foreach (var elem in elems_flat.Skip(1))
                {
                    n_static.merge_with(new Dimension(tensor_shape.dimension_value(elem.shape[0])));
                }

                foreach (Tensor r in results_flat)
                {
                    r.set_shape(new TensorShape(n_static).concatenate(r.shape.Skip(1).ToArray()));
                }

                // todo get working when the above caching_device is fixed
                //if (in_graph_mode && varscope_caching_device_was_none) {
                //    varscope.set_caching_device(None);
                //}

                return output_pack(results_flat);
            });
        }
    }
}

