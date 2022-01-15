using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Framework;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace Tensorflow
{
#pragma warning disable CS0659 // 'Operation' overrides Object.Equals(object o) but does not override Object.GetHashCode()
    public partial class Operation
#pragma warning restore CS0659 // 'Operation' overrides Object.Equals(object o) but does not override Object.GetHashCode()
    {
        /// <summary>
        /// map on the list of tensors unpacked from `elems` on dimension 0.
        /// </summary>
        /// <param name="fn"></param>
        /// <param name="elems"></param>
        /// <param name="dtype"></param>
        /// <param name="parallel_iterations"></param>
        /// <param name="back_prop"></param>
        /// <param name="swap_memory"></param>
        /// <param name="infer_shape"></param>
        /// <param name="name"></param>
        /// <returns>A tensor or (possibly nested) sequence of tensors.</returns>
        public static Tensor map_fn(Func<Tensor, Tensor> fn,
            Tensor elems,
            TF_DataType dtype = TF_DataType.DtInvalid,
            int parallel_iterations = 10,
            bool back_prop = true,
            bool swap_memory = false,
            bool infer_shape = true,
            string name = null)
        {
            bool input_is_sequence = nest.is_sequence(elems);
            Tensor[] input_flatten(Tensor x) => input_is_sequence ? nest.flatten(x).ToArray() : new[] { x };
            Tensor input_pack(Tensor[] x) => input_is_sequence ? (Tensor)nest.pack_sequence_as(elems, x) : x[0];

            bool output_is_sequence;
            Func<Tensor, Tensor[]> output_flatten;
            Func<Tensor[], Tensor> output_pack;
            if (dtype == TF_DataType.DtInvalid)
            {
                output_is_sequence = input_is_sequence;
                output_flatten = input_flatten;
                output_pack = input_pack;
            }
            else
            {
                output_is_sequence = nest.is_sequence(dtype);
                output_flatten = (x) => output_is_sequence ? nest.flatten(x).ToArray() : new[] { x };
                output_pack = (x) => output_is_sequence ? (Tensor)nest.pack_sequence_as(dtype, x) : x[0];
            }

            var elems_flat = input_flatten(elems);
            return tf_with(ops.name_scope(name, "map", elems_flat), delegate
            {
                //if in_graph_mode:
                //# Any get_variable calls in fn will cache the first call locally
                //# and not issue repeated network I/O requests for each iteration.
                //varscope = vs.get_variable_scope()
                //varscope_caching_device_was_none = False
                //if varscope.caching_device is None:
                //  # TODO(ebrevdo): Change to using colocate_with here and in other
                //  # methods.
                //  varscope.set_caching_device(lambda op: op.device)
                //  varscope_caching_device_was_none = True

                elems_flat = elems_flat.Select(elem => ops.convert_to_tensor(elem, name: "elem"))
                    .ToArray();

                dtype = elems_flat.Select(elem => elem.dtype).First();
                var dtype_flat = new[] { dtype };

                // Convert elems to tensor array. n may be known statically.
                var static_shape = elems_flat[0].shape;

                var n = static_shape[0];

                // TensorArrays are always flat
                var elems_ta = elems_flat.Select(elem => tf.TensorArray(dtype: elem.dtype,
                        size: Convert.ToInt32(n),
                        dynamic_size: false,
                        infer_shape: true)).ToArray();

                // Unpack elements
                var elems_ta_1 = new List<TensorArray>();
                foreach (var (elem_ta, elem) in zip(elems_ta, elems_flat))
                    elems_ta_1.Add(elem_ta.unstack(elem));

                elems_ta = elems_ta_1.ToArray();

                var i = constant_op.constant(0);

                var accs_ta = dtype_flat.Select(dt => tf.TensorArray(dtype: dt,
                        size: Convert.ToInt32(n),
                        dynamic_size: false,
                        infer_shape: infer_shape)).ToArray();


                BodyItem compute(BodyItem item)
                {
                    var packed_values = input_pack(elems_ta.Select(elem_ta => elem_ta.read(item.I)).ToArray());
                    var packed_fn_values = fn(packed_values);
                    //nest.assert_same_structure(dtype or elems, packed_fn_values)

                    var flat_fn_values = output_flatten(packed_fn_values);
                    for (int j = 0; j < item.Accs_ta.Length; j++)
                    {
                        item.Accs_ta[j].write(item.I, flat_fn_values[j]);
                    }

                    return new BodyItem(item.I + 1, item.Accs_ta);
                }

                var r_a = control_flow_ops.while_loop(
                    (x) => x.I < n,
                    compute,
                    new BodyItem(i, accs_ta),
                    parallel_iterations: parallel_iterations,
                    back_prop: back_prop,
                    swap_memory: swap_memory,
                    maximum_iterations: tf.constant(n));
                var results_flat = r_a.Accs_ta.Select(r => r.stack()).ToArray();

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
            public TensorArray[] Accs_ta { get; set; }

            public BodyItem()
            {
            }

            public BodyItem(Tensor i, TensorArray[] accs_ta)
            {
                I = i;
                Accs_ta = accs_ta;
            }

            public object[] Flatten()
            {
                var elements = new List<object> { I };
                elements.AddRange(Accs_ta);
                return elements.ToArray();
            }

            public BodyItem Pack(object[] sequences)
            {
                I = sequences[0] as Tensor;
                Accs_ta = new[] { sequences[1] as TensorArray };

                return new BodyItem(I, Accs_ta);
            }

            public BodyItem FromMergeVars(ITensorOrTensorArray[] merge_vars)
            {
                I = (Tensor)merge_vars[1];
                Accs_ta = new[] { (TensorArray)merge_vars[2] };
                return this;
            }
        }
    }
}
