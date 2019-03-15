using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations;

namespace Tensorflow.Gradients
{
    public class nn_grad
    {
        /// <summary>
        /// Return the gradients for the 2 inputs of bias_op.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="grad"></param>
        /// <returns></returns>
        public static Tensor[] _BiasAddGrad(Operation op, Tensor grad)
        {
            string data_format = op.get_attr("data_format")?.ToString();
            var bias_add_grad = gen_nn_ops.bias_add_grad(out_backprop: grad, data_format: data_format);
            return new Tensor[] { grad, bias_add_grad };
        }

        public static Tensor[] _ReluGrad(Operation op, Tensor grad)
        {
            return new Tensor[] { gen_nn_ops.relu_grad(grad, op.outputs[0]) };
        }

        /// <summary>
        /// Gradient function for SoftmaxCrossEntropyWithLogits.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="grad_loss"></param>
        /// <param name="grad_grad"></param>
        /// <returns></returns>
        public static Tensor[] _SoftmaxCrossEntropyWithLogitsGrad(Operation op, Tensor[] grads)
        {
            var grad_loss = grads[0];
            var grad_grad = grads[1];
            var softmax_grad = op.outputs[1];

            throw new NotImplementedException("_SoftmaxCrossEntropyWithLogitsGrad");
        }
    }
}
