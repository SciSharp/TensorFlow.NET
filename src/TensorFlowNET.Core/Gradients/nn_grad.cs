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
using System.Linq;
using Tensorflow.Operations;

namespace Tensorflow.Gradients
{
    /// <summary>
    /// 
    /// </summary>
    [RegisterGradient("math_grad")]
    public class nn_grad
    {
        /// <summary>
        /// Return the gradients for the 2 inputs of bias_op.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="grads"></param>
        /// <returns></returns>
        [RegisterGradient("BiasAdd")]
        public static Tensor[] _BiasAddGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            string data_format = op.get_attr("data_format")?.ToString();
            var bias_add_grad = gen_nn_ops.bias_add_grad(out_backprop: grad, data_format: data_format);
            return new Tensor[] { grad, bias_add_grad };
        }

        [RegisterGradient("Relu")]
        public static Tensor[] _ReluGrad(Operation op, Tensor[] grads)
        {
            return new Tensor[] { gen_nn_ops.relu_grad(grads[0], op.outputs[0]) };
        }

        [RegisterGradient("LeakyRelu")]
        public static Tensor[] _LeakyReluGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var x = op.inputs[0];
            var alpha = (float)op.get_attr("alpha");
            return new Tensor[] { gen_nn_ops.leaky_relu_grad(grad, x, alpha: alpha) };
        }

        /// <summary>
        /// The derivative of the softmax nonlinearity.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="grads"></param>
        /// <returns></returns>
        [RegisterGradient("Softmax")]
        public static Tensor[] _SoftmaxGrad(Operation op, Tensor[] grads)
        {
            var grad_softmax = grads[0];

            var softmax = op.outputs[0];
            var mul = grad_softmax * softmax;
            var sum_channels = math_ops.reduce_sum(mul, -1, keepdims: true);
            var sub = grad_softmax - sum_channels;
            return new Tensor[] { sub * softmax };
        }

        /// <summary>
        /// Gradient function for SoftmaxCrossEntropyWithLogits.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="grads"></param>
        /// <returns></returns>
        [RegisterGradient("SoftmaxCrossEntropyWithLogits")]
        public static Tensor[] _SoftmaxCrossEntropyWithLogitsGrad(Operation op, Tensor[] grads)
        {
            var grad_loss = grads[0];
            var grad_grad = grads[1];
            var softmax_grad = op.outputs[1];
            var grad = _BroadcastMul(grad_loss, softmax_grad);

            var logits = op.inputs[0];
            if (grad_grad != null && !IsZero(grad_grad))
            {
                throw new NotImplementedException("_SoftmaxCrossEntropyWithLogitsGrad");
            }

            return new Tensor[]
            {
                grad,
                _BroadcastMul(grad_loss, -nn_ops.log_softmax(logits))
            };
        }

        [RegisterGradient("SparseSoftmaxCrossEntropyWithLogits")]
        public static Tensor[] _SparseSoftmaxCrossEntropyWithLogitsGrad(Operation op, Tensor[] grads)
        {
            var sparse_softmax_grad_without_gradient = array_ops.prevent_gradient(
              op.outputs[1],
              message: "Currently there is no way to take the second " +
              "derivative of sparse_softmax_cross_entropy_with_logits due to the fused " +
              "implementation's interaction with tf.gradients()");

            var grad_0 = grads[0];
            
            return new Tensor[]
            {
                _BroadcastMul(grad_0, sparse_softmax_grad_without_gradient),
                null
            };
        }

        [RegisterGradient("SquaredDifference")]
        public static Tensor[] _SquaredDifferenceGrad(Operation op, Tensor[] grads)
        {
           //"""Returns the gradient for (x-y)^2."""
            Tensor x = op.inputs[0];
            Tensor y = op.inputs[1];
            return new Tensor[]
            {
                x,
                y
            };
        }
        /// <summary>
        /// Gradient function for Conv2D.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="grads"></param>
        /// <returns></returns>
        [RegisterGradient("Conv2D")]
        public static Tensor[] _Conv2DGrad(Operation op, Tensor[] grads)
        {
            var dilations = op.get_attr_list<int>("dilations");
            var strides = op.get_attr_list<int>("strides");
            var padding = op.get_attr<string>("padding");
            var explicit_paddings = op.get_attr_list<int>("explicit_paddings");
            var use_cudnn_on_gpu = op.get_attr<bool>("use_cudnn_on_gpu");
            var data_format = op.get_attr<string>("data_format");
            var shape = gen_array_ops.shape_n(new Tensor[] { op.inputs[0], op.inputs[1] });

            return new Tensor[]
            {
                gen_nn_ops.conv2d_backprop_input(shape[0], op.inputs[1], grads[0],
                    strides, padding, use_cudnn_on_gpu, explicit_paddings,
                    dilations: dilations,
                    data_format: data_format),
                gen_nn_ops.conv2d_backprop_filter(op.inputs[0], shape[1], grads[0],
                    strides, padding,
                    dilations: dilations,
                    explicit_paddings: explicit_paddings,
                    use_cudnn_on_gpu: use_cudnn_on_gpu,
                    data_format: data_format)
            };
        }

        [RegisterGradient("FusedBatchNorm")]
        public static Tensor[] _FusedBatchNormGrad(Operation op, Tensor[] grads)
            => _BaseFusedBatchNormGrad(op, 0, grads);

        [RegisterGradient("FusedBatchNormV2")]
        public static Tensor[] _FusedBatchNormV2Grad(Operation op, Tensor[] grads)
            => _BaseFusedBatchNormGrad(op, 1, grads);

        [RegisterGradient("FusedBatchNormV3")]
        public static Tensor[] _FusedBatchNormV3Grad(Operation op, Tensor[] grads)
            => _BaseFusedBatchNormGrad(op, 2, grads);

        /// <summary>
        /// Return the gradients for the 3 inputs of BatchNorm.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="version"></param>
        /// <param name="grads"></param>
        /// <returns></returns>
        public static Tensor[] _BaseFusedBatchNormGrad(Operation op, int version, Tensor[] grads)
        {
            var x = op.inputs[0];
            var grad_y = grads[0];
            var scale = op.inputs[1];
            var epsilon = op.get_attr<float>("epsilon");
            var data_format = op.get_attr<string>("data_format");
            var is_training = op.get_attr<bool>("is_training");
            Func<FusedBatchNormParams, Tensor[]> grad_fun = null;

            switch (version)
            {
                case 2:
                    grad_fun = gen_nn_ops.fused_batch_norm_grad_v3;
                    break;
                case 1:
                    // grad_fun = gen_nn_ops.fused_batch_norm_grad_v2;
                    throw new NotImplementedException("");
                default:
                    grad_fun = gen_nn_ops.fused_batch_norm_grad;
                    break;
            }

            if (is_training)
            {
                return grad_fun(new FusedBatchNormParams
                {
                    YBackprop = grad_y,
                    X = x,
                    Scale = scale,
                    ReserveSpace1 = op.outputs[3],
                    ReserveSpace2 = op.outputs[4],
                    ReserveSpace3 = version == 2 ? op.outputs[5] : null,
                    Epsilon = epsilon,
                    DataFormat = data_format,
                    IsTraining = is_training
                });
            }
            else
            {
                var pop_mean = op.inputs[3];
                var pop_var = op.inputs[4];
                if (data_format == "NCHW")
                    throw new NotImplementedException("");

                var results = grad_fun(new FusedBatchNormParams
                {
                    YBackprop = grad_y,
                    X = x,
                    Scale = scale,
                    ReserveSpace1 = pop_mean,
                    ReserveSpace2 = pop_var,
                    ReserveSpace3 = version == 2 ? op.outputs[5] : null,
                    Epsilon = epsilon,
                    DataFormat = data_format,
                    IsTraining = is_training
                });

                var (dx, dscale, doffset) = (results[0], results[1], results[2]);
                if (data_format == "NCHW")
                    throw new NotImplementedException("");

                return new Tensor[]
                {
                    dx,
                    dscale,
                    doffset,
                    null,
                    null
                };
            }
        }

        [RegisterGradient("BatchNormWithGlobalNormalization")]
        public static Tensor _BatchNormWithGlobalNormalizationGrad(Operation op, Tensor[] grads)
        {
            throw new NotImplementedException("BatchNormWithGlobalNormalization");
        }

        private static bool IsZero(Tensor g)
        {
            if (new string[] { "ZerosLike", "Zeros" }.Contains(g.op.type))
                return true;

            throw new NotImplementedException("IsZero");
        }

        private static Tensor _BroadcastMul(Tensor vec, Tensor mat)
        {
            vec = array_ops.expand_dims(vec, -1);
            return vec * mat;
        }

        [RegisterGradient("MaxPool")]
        public static Tensor[] _MaxPoolGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            return new Tensor[]
            {
                gen_nn_ops.max_pool_grad(
                  op.inputs[0],
                  op.outputs[0],
                  grad,
                  op.get_attr_list<int>("ksize"),
                  op.get_attr_list<int>("strides"),
                  padding: op.get_attr("padding").ToString(),
                  data_format: op.get_attr("data_format").ToString())
            };
        }

        /// <summary>
        /// Return the gradients for TopK.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="grads"></param>
        /// <returns></returns>
        [RegisterGradient("TopK")]
        public static Tensor[] _TopKGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var _ = grads[1];

            var in_shape = array_ops.shape(op.inputs[0]);
            var ind_shape = array_ops.shape(op.outputs[1]);

            // int32 is not supported on GPU hence up-casting
            var cast = math_ops.cast(ind_shape, TF_DataType.TF_INT64);
            var size = array_ops.size(ind_shape) - 1;
            var ind_lastdim = array_ops.gather(cast, size);

            // Flatten indices to 2D.
            var stack = array_ops.stack(new object[] { -1L, ind_lastdim });
            var ind_2d = array_ops.reshape(op.outputs[1], stack);

            var in_lastdim = array_ops.gather(math_ops.cast(in_shape, TF_DataType.TF_INT64),
                array_ops.size(in_shape) - 1);
            var outerdim = array_ops.shape(ind_2d).slice(0);

            // Compute linear indices(flattened to 1D).
            var cast1 = math_ops.cast(outerdim, TF_DataType.TF_INT64);
            var range2 = math_ops.range(0L, cast1 * in_lastdim, in_lastdim);
            var dim2 = array_ops.expand_dims(range2, -1);
            var cast2 = math_ops.cast(dim2, TF_DataType.TF_INT32);
            var ind = array_ops.reshape(ind_2d + cast2, new int[] { -1 });

            // Substitute grad to appropriate locations and fill the rest with zeros,
            // finally reshaping it to the original input shape.
            var scatter = gen_array_ops.scatter_nd(array_ops.expand_dims(ind, -1),
              array_ops.reshape(grad, new int[] { -1 }),
              new Tensor[] { math_ops.reduce_prod(in_shape) });

            return new Tensor[]
            {
                array_ops.reshape(scatter, in_shape),
                array_ops.zeros(new int[0], dtype: TF_DataType.TF_INT32)
            };
        }
    }
}
