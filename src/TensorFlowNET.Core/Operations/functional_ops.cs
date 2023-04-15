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

using Google.Protobuf;
using Google.Protobuf.WellKnownTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Framework;
using Tensorflow.Functions;
using Tensorflow.Operations;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class functional_ops
    {
        public static Tensor[] partitioned_call(Tensors args, EagerDefinedFunction f, DataType[] tout,
            bool executing_eagerly, string config, string executor_type)
        {
            if (tout is null)
            {
                throw new NotImplementedException();
            }

            if (config is null)
            {
                config = function_utils.get_disabled_rewriter_config().ToStringUtf8();
            }

            if (executor_type is null)
            {
                executor_type = "";
            }

            if (executing_eagerly)
            {
                // TODO(Rinne): implement it.
                
                throw new NotImplementedException();
            }

            var converted_args = args.Select(x => ops.convert_to_tensor(x)).ToArray();
            AttrValue tin_attr = new()
            {
                List = new AttrValue.Types.ListValue()
            };
            tin_attr.List.Type.AddRange(args.Select(x => x.dtype.as_datatype_enum()));
            AttrValue tout_attr = new()
            {
                List = new AttrValue.Types.ListValue()
            };
            tout_attr.List.Type.AddRange(tout);
            AttrValue func_attr = new()
            {
                Func = new NameAttrList()
            };
            func_attr.Func.Name = f.Name;
            AttrValue executor_type_attr = new AttrValue()
            {
                S = tf.compat.as_bytes(executor_type)
            };
            AttrValue config_proto = new AttrValue()
            {
                S = ByteString.CopyFromUtf8(executor_type)
            };

            var graph = ops.get_default_graph();
            f.AddToGraph(graph);
            // TODO(Rinne): complete it with `f.stateful`
            var op_name = "PartitionedCall";
            string xla_compile_attr = "_XlaMustCompile";
            Dictionary<string, AttrValue> op_attrs = new();
            op_attrs["Tin"] = tin_attr;
            op_attrs["Tout"] = tout_attr;
            op_attrs["f"] = func_attr;
            op_attrs["config_proto"] = config_proto;
            op_attrs["executor_type"] = executor_type_attr;
            // TODO(Rinne): deal with `f.definition`.
            var op = graph.create_op(op_name, args, tout.Select(x => x.as_tf_dtype()).ToArray(), 
                name: op_name, attrs: op_attrs);
            var outputs = op.outputs;
            // TODO(Rinne): deal with `f.graph`.
            return outputs;
        }
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

            Tensor[] input_flatten(Tensor x) => input_is_sequence ? nest.flatten(x).ToArray() : new[] { x };
            Tensor input_pack(Tensor[] x) => input_is_sequence ? (Tensor)nest.pack_sequence_as(elems, x) : x[0];

            bool output_is_sequence;
            Func<Tensor, Tensor[]> output_flatten;
            Func<Tensor[], Tensor> output_pack;
            if (initializer == null)
            {
                output_is_sequence = input_is_sequence;
                output_flatten = input_flatten;
                output_pack = input_pack;
            }
            else
            {
                output_is_sequence = nest.is_sequence(initializer);
                output_flatten = (x) => output_is_sequence ? nest.flatten(x).ToArray() : new[] { x };
                output_pack = (x) => output_is_sequence ? (Tensor)nest.pack_sequence_as(initializer, x) : x[0];
            }

            var elems_flat = input_flatten(elems);

            bool in_graph_mode = tf.Context.executing_eagerly();

            return tf_with(ops.name_scope(name, "scan", new { elems_flat }), scope =>
            {
                if (in_graph_mode)
                {
                    // todo tf.net doesn't expose .caching_device
                    //// Any get_variable calls in fn will cache the first call locally
                    //// and not issue repeated network I/O requests for each iteration.
                    //var varscope = variable_scope.get_variable_scope();
                    //bool varscope_caching_device_was_none = false;
                    //if (varscope.caching_device = null)
                    //{
                    //    //      varscope.set_caching_device(lambda op: op.device)
                    //    //      varscope_caching_device_was_none = True
                    //}
                }

                elems_flat = elems_flat.Select(elem => ops.convert_to_tensor(elem, name: "elem")).ToArray();

                var n = tensor_shape.dimension_value(elems_flat[0].shape[0]);

                // todo python had the below but dimension_value returns int which can't be null
                //if (n == null)
                //{
                //    n = array_ops.shape(elems_flat[0])[0];
                //}

                var elems_ta = elems_flat.Select(elem => tf.TensorArray(
                    elem.dtype,
                    size: n,
                    dynamic_size: false,
                    element_shape: elem.shape.dims.Skip(1).ToArray(),
                    infer_shape: true)).ToList();

                for (int index = 0; index < elems_ta.Count; index++)
                {
                    elems_ta[index].unstack(elems_flat[index]);
                }

                Tensor[] a_flat;
                int i;
                if (initializer == null)
                {
                    a_flat = elems_ta.Select(elem => elem.read(tf.constant(reverse ? n - 1 : 0))).ToArray();
                    i = 1;
                }
                else
                {
                    Tensor[] initializer_flat = output_flatten(initializer);
                    a_flat = initializer_flat.Select(init => ops.convert_to_tensor(init)).ToArray();
                    i = 0;
                }

                var accs_ta = a_flat.Select(init => tf.TensorArray(
                    dtype: init.dtype,
                    size: n,
                    element_shape: infer_shape ? init.shape : null,
                    dynamic_size: false,
                    infer_shape: infer_shape)).ToArray();

                if (initializer == null)
                {
                    for (int index = 0; index < accs_ta.Length; index++)
                    {
                        accs_ta[index].write(reverse ? n - 1 : 0, a_flat[index]);
                    }
                }

                BodyItem compute(BodyItem item)
                {
                    var packed_elems = input_pack(elems_ta.Select(elem_ta => elem_ta.read(item.I)).ToArray());
                    var packed_a = output_pack(item.A_Flat);
                    var a_out = fn(packed_a, packed_elems);

                    var flat_a_out = output_flatten(a_out);
                    for (int j = 0; j < item.Accs_ta.Length; j++)
                    {
                        item.Accs_ta[j].write(item.I, flat_a_out[j]);
                    }

                    var next_i = reverse ? item.I - 1 : item.I + 1;
                    return new BodyItem(next_i, flat_a_out, item.Accs_ta);
                }

                int initial_i;
                Func<BodyItem, Tensor> condition;
                if (reverse)
                {
                    initial_i = n - 1 - i;
                    condition = x => x.I >= 0;
                }
                else
                {
                    initial_i = i;
                    condition = x => x.I < n;
                }

                BodyItem bodyItem =
                    control_flow_ops.while_loop(
                        condition,
                        compute,
                        new BodyItem(tf.constant(initial_i), a_flat, accs_ta),
                        parallel_iterations: parallel_iterations,
                        back_prop: back_prop,
                        swap_memory: swap_memory,
                        maximum_iterations: tf.constant(n));

                var results_flat = bodyItem.Accs_ta.Select(r => r.stack()).ToArray();

                var n_static = new Dimension(tensor_shape.dimension_value(elems_flat[0].shape.with_rank_at_least(1).dims[0]));

                foreach (var elem in elems_flat.Skip(1))
                {
                    n_static.merge_with(new Dimension(tensor_shape.dimension_value(elem.shape.with_rank_at_least(1).dims[0])));
                }

                foreach (Tensor r in results_flat)
                {
                    r.shape = new Shape(n_static).concatenate(r.dims.Skip(1).ToArray());
                }

                // todo get working when the above caching_device is fixed
                //if (in_graph_mode && varscope_caching_device_was_none) {
                //    varscope.set_caching_device(None);
                //}

                return output_pack(results_flat);
            });
        }

        internal class BodyItem : ICanBeFlattened, IPackable<BodyItem>, IFromMergeVars<BodyItem>
        {
            public Tensor I { get; set; }
            public Tensor[] A_Flat { get; set; }
            public TensorArray[] Accs_ta { get; set; }

            public BodyItem()
            {
            }

            public BodyItem(Tensor i, Tensor[] a_flat, TensorArray[] accs_ta)
            {
                I = i;
                A_Flat = a_flat;
                Accs_ta = accs_ta;
            }

            public object[] Flatten()
            {
                var elements = new List<object> { I };
                elements.AddRange(A_Flat);
                elements.AddRange(Accs_ta);
                return elements.ToArray();
            }

            public BodyItem Pack(object[] sequences)
            {
                I = sequences[0] as Tensor;
                A_Flat = new[] { sequences[1] as Tensor };
                Accs_ta = new[] { sequences[2] as TensorArray };

                return new BodyItem(I, A_Flat, Accs_ta);
            }

            public BodyItem FromMergeVars(ITensorOrTensorArray[] merge_vars)
            {
                I = (Tensor)merge_vars[1];
                A_Flat = new[] { (Tensor)merge_vars[2] };
                Accs_ta = new[] { (TensorArray)merge_vars[3] };
                return this;
            }
        }
    }
}

