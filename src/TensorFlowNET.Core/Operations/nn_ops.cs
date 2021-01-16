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
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Operations;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class nn_ops
    {
        public static ConvolutionInternal convolution_internal(string padding,
            int[] strides,
            int[] dilation_rate,
            string name = null,
            string data_format = null) => new ConvolutionInternal(new ConvolutionalArgs
            {
                Padding = padding,
                Strides = strides,
                DilationRate = dilation_rate,
                DataFormat = data_format,
                Name = name
            });

        /// <summary>
        /// Adds `bias` to `value`.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="bias"></param>
        /// <param name="data_format"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor bias_add(Tensor value,
            IVariableV1 bias,
            string data_format = null,
            string name = null)
        {
            return tf_with(ops.name_scope(name, "BiasAdd", new { value, bias }), scope =>
            {
                name = scope;
                return gen_nn_ops.bias_add(value, bias, data_format: data_format, name: name);
            });
        }

        /// <summary>
        /// Computes dropout.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="rate"></param>
        /// <param name="noise_shape"></param>
        /// <param name="seed"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor dropout_v2(Tensor x, Tensor rate, Tensor noise_shape = null, int? seed = null, string name = null)
        {
            return tf_with(ops.name_scope(name, "dropout", x), scope =>
            {
                name = scope;
                x = ops.convert_to_tensor(x, name: "x");
                if (!x.dtype.is_floating())
                    throw new NotImplementedException($"x has to be a floating point tensor since it's going to" +
                        $" be scaled. Got a {x.dtype} tensor instead.");

                var keep_prob = 1 - rate;
                var scale = 1 / keep_prob;
                var scale_tensor = ops.convert_to_tensor(scale, dtype: x.dtype);
                var ret = gen_math_ops.mul(x, scale_tensor);

                noise_shape = _get_noise_shape(x, noise_shape);

                // Sample a uniform distribution on [0.0, 1.0) and select values larger than
                // rate.
                //
                // NOTE: Random uniform actually can only generate 2^23 floats on [1.0, 2.0)
                // and subtract 1.0.
                var random_tensor = random_ops.random_uniform(noise_shape, seed: seed, dtype: x.dtype);
                // NOTE: if (1.0 + rate) - 1 is equal to rate, then we want to consider that
                // float to be selected, hence we use a >= comparison.
                var keep_mask = random_tensor >= rate;
                ret = x * scale * math_ops.cast(keep_mask, x.dtype);
                if (!tf.executing_eagerly())
                    ret.set_shape(x.TensorShape);
                return ret;
            });
        }

        private static Tensor _get_noise_shape(Tensor x, Tensor noise_shape)
        {
            if (noise_shape == null)
                return array_ops.shape(x);
            else
                return noise_shape;
        }

        public static Tensor in_top_k(Tensor predictions, Tensor targets, int k, string name = null)
        {
            return tf_with(ops.name_scope(name, "in_top_k"), delegate
            {
                return gen_nn_ops.in_top_kv2(predictions, targets, k, name: name);
            });
        }

        public static Tensor log_softmax(Tensor logits, int axis = -1, string name = null)
        {
            return _softmax(logits, gen_nn_ops.log_softmax, axis, name);
        }

        /// <param name="axis">equivalent to `dim`</param>
        public static Tensor softmax(Tensor logits, int axis = -1, string name = null)
        {
            return _softmax(logits, gen_nn_ops.softmax, axis, name);
        }

        public static Tensor leaky_relu(Tensor features, float alpha = 0.2f, string name = null)
        {
            return tf_with(ops.name_scope(name, "LeakyRelu", new { features, alpha }), scope =>
            {
                name = scope;
                features = ops.convert_to_tensor(features, name: "features");
                if (features.dtype.is_integer())
                    features = math_ops.cast(features, dtypes.float32);
                return gen_nn_ops.leaky_relu(features, alpha: alpha, name: name);
                //return math_ops.maximum(alpha * features, features, name: name);
            });
        }

        /// <summary>
        /// Performs the max pooling on the input.
        /// </summary>
        /// <param name="value">A 4-D `Tensor` of the format specified by `data_format`.</param>
        /// <param name="ksize">
        /// A list or tuple of 4 ints. The size of the window for each dimension
        /// of the input tensor.
        /// </param>
        /// <param name="strides">
        /// A list or tuple of 4 ints. The stride of the sliding window for
        /// each dimension of the input tensor.
        /// </param>
        /// <param name="padding">A string, either `'VALID'` or `'SAME'`. The padding algorithm.</param>
        /// <param name="data_format">A string. 'NHWC', 'NCHW' and 'NCHW_VECT_C' are supported.</param>
        /// <param name="name">Optional name for the operation.</param>
        /// <returns></returns>
        public static Tensor max_pool(Tensor value, int[] ksize, int[] strides, string padding, string data_format = "NHWC", string name = null)
        {
            return tf_with(ops.name_scope(name, "MaxPool", value), scope =>
            {
                name = scope;
                value = ops.convert_to_tensor(value, name: "input");
                return gen_nn_ops.max_pool(
                    value,
                    ksize: ksize,
                    strides: strides,
                    padding: padding,
                    data_format: data_format,
                    name: name);
            });
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

        /// <summary>
        /// Computes sparse softmax cross entropy between `logits` and `labels`.
        /// </summary>
        /// <param name="labels"></param>
        /// <param name="logits"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor sparse_softmax_cross_entropy_with_logits(Tensor labels = null,
            Tensor logits = null, string name = null)
        {
            // Reshape logits and labels to rank 2.
            return tf_with(ops.name_scope(name, default_name: "SparseSoftmaxCrossEntropyWithLogits", (labels, logits)), delegate
            {
                labels = ops.convert_to_tensor(labels);
                logits = ops.convert_to_tensor(logits);
                var precise_logits = logits.dtype == TF_DataType.TF_HALF ? math_ops.cast(logits, dtypes.float32) : logits;

                // Store label shape for result later.
                var labels_static_shape = labels.TensorShape;
                var labels_shape = array_ops.shape(labels);
                /*bool static_shapes_fully_defined = (
                    labels_static_shape.is_fully_defined() &&
                        logits.get_shape()[:-1].is_fully_defined());*/

                // Check if no reshapes are required.
                if (logits.TensorShape.ndim == 2)
                {
                    var (cost, _) = gen_nn_ops.sparse_softmax_cross_entropy_with_logits(
                        precise_logits, labels, name: name);
                    if (logits.dtype == dtypes.float16)
                        return math_ops.cast(cost, dtypes.float32);
                    else
                        return cost;
                }

                // Perform a check of the dynamic shapes if the static shapes are not fully
                // defined.
                throw new NotImplementedException("sparse_softmax_cross_entropy_with_logits");
            });
        }

        public static Tensor softmax_cross_entropy_with_logits_v2_helper(Tensor labels,
            Tensor logits,
            int axis = -1,
            string name = null)
        {
            return tf_with(ops.name_scope(name, "softmax_cross_entropy_with_logits", new { logits, labels }), scope =>
            {
                name = scope;
                var precise_logits = logits;
                var input_rank = array_ops.rank(precise_logits);
                var shape = logits.TensorShape;

                if (axis != -1)
                    throw new NotImplementedException("softmax_cross_entropy_with_logits_v2_helper axis != -1");

                var input_shape = array_ops.shape(precise_logits);

                // Make precise_logits and labels into matrices.
                precise_logits = _flatten_outer_dims(precise_logits);
                labels = _flatten_outer_dims(labels);

                // Do the actual op computation.
                // The second output tensor contains the gradients.  We use it in
                // _CrossEntropyGrad() in nn_grad but not here.

                var (cost, unused_backprop) = gen_nn_ops.softmax_cross_entropy_with_logits(precise_logits, labels, name: name);

                // The output cost shape should be the input minus axis.
                var output_shape = array_ops.slice(input_shape,
                    new Tensor[] { constant_op.constant(0) },
                    new Tensor[] { math_ops.subtract(input_rank, 1) });

                cost = array_ops.reshape(cost, output_shape);

                return cost;
            });
        }

        /// <summary>
        /// Flattens logits' outer dimensions and keep its last dimension.
        /// </summary>
        /// <param name="logits"></param>
        /// <returns></returns>
        private static Tensor _flatten_outer_dims(Tensor logits)
        {
            var rank = array_ops.rank(logits);
            var last_dim_size = array_ops.slice(array_ops.shape(logits),
                new[] { math_ops.subtract(rank, 1) },
                new[] { constant_op.constant(1) });

            var ops = array_ops.concat(new[] { new[] { -1 }, (object)last_dim_size }, 0);
            var output = array_ops.reshape(logits, ops);

            // Set output shape if known.
            if (!tf.Context.executing_eagerly())
            {
                var shape = logits.TensorShape;
                if (shape != null && shape.ndim > 0)
                {
                    var product = 1;
                    var product_valid = true;
                    foreach (var d in shape.dims.Take(shape.ndim - 1))
                    {
                        if (d == -1)
                        {
                            product_valid = false;
                            break;
                        }
                        else
                        {
                            product *= d;
                        }
                    }

                    if (product_valid)
                    {
                        var output_shape = new[] { product };
                        throw new NotImplementedException("_flatten_outer_dims product_valid");
                    }
                }
            }

            return output;
        }
    }
}
