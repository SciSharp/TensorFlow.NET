using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Operations;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class Operation
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
            var elems_flat = new[] { elems };
            tf_with(ops.name_scope(name, "map", elems_flat), delegate
            {
                var varscope = tf.get_variable_scope();
                elems_flat = elems_flat.Select(elem => ops.convert_to_tensor(elem, name: "elem"))
                    .ToArray();

                dtype = elems_flat.Select(elem => elem.dtype).First();
                var dtype_flat = new[] { dtype };

                // Convert elems to tensor array. n may be known statically.
                var static_shape = elems_flat[0].shape;

                var n = static_shape[0];

                // TensorArrays are always flat
                var elems_ta = elems_flat.Select(elem => new TensorArray(dtype: elem.dtype,
                        size: ops.convert_to_tensor(n),
                        dynamic_size: false,
                        infer_shape: true)).ToArray();

                // Unpack elements
                var elems_ta_1 = new List<TensorArray>();
                foreach (var (elem_ta, elem) in zip(elems_ta, elems_flat))
                    elems_ta_1.Add(elem_ta.unstack(elem));

                elems_ta = elems_ta_1.ToArray();

                var i = constant_op.constant(0);

                var accs_ta = dtype_flat.Select(dt => new TensorArray(dtype: dt,
                        size: ops.convert_to_tensor(n),
                        dynamic_size: false,
                        infer_shape: infer_shape)).ToArray();

                /*Func<Tensor, TensorArray> compute = (i, tas) =>
                {
                    throw new NotImplementedException("");
                };

                var r_a = control_flow_ops.while_loop(
                    (i, _) => i < n, 
                    compute, 
                    new[] { i, accs_ta },
                    parallel_iterations: parallel_iterations,
                    back_prop: back_prop,
                    swap_memory: swap_memory,
                    maximum_iterations: n);*/
            });

            throw new NotImplementedException("");
        }
    }
}
