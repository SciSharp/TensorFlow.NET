using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations;

namespace Tensorflow
{
    public class nn_ops
    {
        public static Convolution Convolution(TensorShape input_shape,
            TensorShape filter_shape,
            string padding,
            int[] strides,
            int[] dilation_rate,
            string name = null,
            string data_format = null) => new Convolution(input_shape,
                filter_shape,
                padding,
                strides,
                dilation_rate,
                name: name,
                data_format: data_format);

        /// <summary>
        /// Adds `bias` to `value`.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="bias"></param>
        /// <param name="data_format"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor bias_add(Tensor value, 
            Tensor bias, 
            string data_format = null, 
            string name = null)
        {
            return Python.with(ops.name_scope(name, "BiasAdd", new { value, bias }), scope =>
            {
                value = ops.convert_to_tensor(value, name: "input");
                var bias_tensor = ops.convert_to_tensor(bias, dtype: value.dtype, name: "bias");
                return gen_nn_ops.bias_add(value, bias_tensor, data_format: data_format, name: name);
            });
        }

        public static Tensor log_softmax(Tensor logits, int axis = -1, string name = null)
        {
            return _softmax(logits, gen_nn_ops.log_softmax, axis, name);
        }

        public static Tensor _softmax(Tensor logits, Func<Tensor, string, Tensor> compute_op, int dim = -1, string name = null)
        {
            logits = ops.convert_to_tensor(logits);

            var shape = logits.shape;
            bool is_last_dim = dim == -1 || dim == shape.Length - 1;
            if (is_last_dim)
                return compute_op(logits, name);

            throw new NotImplementedException("_softmax helper");
        }

        public static Tensor softmax_cross_entropy_with_logits_v2_helper(Tensor labels,
            Tensor logits,
            int axis = -1,
            string name = null)
        {
            return Python.with(ops.name_scope(name, "softmax_cross_entropy_with_logits", new { }), scope =>
            {
                var precise_logits = logits;
                var input_rank = array_ops.rank(precise_logits);
                var shape = logits.getShape();

                if (axis != -1)
                    throw new NotImplementedException("softmax_cross_entropy_with_logits_v2_helper axis != -1");

                var input_shape = array_ops.shape(precise_logits);

                // Do the actual op computation.
                // The second output tensor contains the gradients.  We use it in
                // _CrossEntropyGrad() in nn_grad but not here.

                var (cost, unused_backprop) = gen_nn_ops.softmax_cross_entropy_with_logits(precise_logits, labels, name: name);

                // The output cost shape should be the input minus axis.
                var output_shape = array_ops.slice(input_shape, 
                    new int[] { 0 },
                    new Tensor[] { math_ops.subtract(input_rank, 1) });

                cost = array_ops.reshape(cost, output_shape);

                return cost;
            });
        }
    }
}
