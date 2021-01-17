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

using System.Linq;
using static Tensorflow.Binding;

namespace Tensorflow.Operations
{
    public class gen_nn_ops
    {
        /// <summary>
        /// Computes a 2-D convolution given 4-D `input` and `filter` tensors.
        /// 
        /// Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
        /// and a filter / kernel tensor of shape
        /// `[filter_height, filter_width, in_channels, out_channels]`, this op
        /// performs the following:
        /// 
        /// 1. Flattens the filter to a 2-D matrix with shape
        ///    `[filter_height * filter_width * in_channels, output_channels]`.
        /// 2. Extracts image patches from the input tensor to form a *virtual*
        ///    tensor of shape `[batch, out_height, out_width,
        ///    filter_height * filter_width * in_channels]`.
        /// 3. For each patch, right-multiplies the filter matrix and the image patch
        ///    vector.
        /// </summary>
        /// <param name="parameters"></param>
        /// <returns></returns>
        public static Tensor conv2d(Conv2dParams parameters)
        {
            if (tf.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "Conv2D", parameters.Name,
                    null,
                    parameters.Input, parameters.Filter,
                    "strides", parameters.Strides,
                    "use_cudnn_on_gpu", parameters.UseCudnnOnGpu,
                    "padding", parameters.Padding,
                    "explicit_paddings", parameters.ExplicitPaddings,
                    "data_format", parameters.DataFormat,
                    "dilations", parameters.Dilations);

                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("Conv2D", name: parameters.Name, args: new
            {
                input = parameters.Input,
                filter = parameters.Filter,
                strides = parameters.Strides,
                padding = parameters.Padding,
                use_cudnn_on_gpu = parameters.UseCudnnOnGpu,
                explicit_paddings = parameters.ExplicitPaddings,
                data_format = parameters.DataFormat,
                dilations = parameters.Dilations
            });

            return _op.outputs[0];
        }

        /// <summary>
        /// Computes the gradients of convolution with respect to the filter.
        /// </summary>
        /// <param name="parameters"></param>
        /// <returns></returns>
        public static Tensor conv2d_backprop_filter(Tensor input, Tensor filter_sizes, Tensor out_backprop,
            int[] strides, string padding, bool use_cudnn_on_gpu = true,
            int[] explicit_paddings = null,
            string data_format = "NHWC",
            int[] dilations = null,
            string name = null)
        {
            if (explicit_paddings == null)
                explicit_paddings = new int[0];
            if (dilations == null)
                dilations = new int[] { 1, 1, 1, 1 };

            if (tf.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "Conv2DBackpropFilter", name,
                    null,
                    input, filter_sizes, out_backprop,
                    "strides", strides,
                    "use_cudnn_on_gpu", use_cudnn_on_gpu,
                    "padding", padding,
                    "explicit_paddings", explicit_paddings,
                    "data_format", data_format,
                    "dilations", dilations);

                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("Conv2DBackpropFilter", name: name, args: new
            {
                input,
                filter_sizes,
                out_backprop,
                strides,
                padding,
                use_cudnn_on_gpu,
                explicit_paddings,
                data_format,
                dilations
            });

            return _op.outputs[0];
        }

        /// <summary>
        /// Computes the gradients of convolution with respect to the input.
        /// </summary>
        /// <param name="parameters"></param>
        /// <returns></returns>
        public static Tensor conv2d_backprop_input(Tensor input_sizes, Tensor filter, Tensor out_backprop,
            int[] strides, string padding, bool use_cudnn_on_gpu = true,
            int[] explicit_paddings = null,
            string data_format = "NHWC",
            int[] dilations = null,
            string name = null)
        {
            if (explicit_paddings == null)
                explicit_paddings = new int[0];
            if (dilations == null)
                dilations = new int[] { 1, 1, 1, 1 };

            if (tf.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "Conv2DBackpropInput", name,
                    null,
                    input_sizes, filter, out_backprop,
                    "strides", strides,
                    "use_cudnn_on_gpu", use_cudnn_on_gpu,
                    "padding", padding,
                    "explicit_paddings", explicit_paddings,
                    "data_format", data_format,
                    "dilations", dilations);

                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("Conv2DBackpropInput", name: name, args: new
            {
                input_sizes,
                filter,
                out_backprop,
                strides,
                padding,
                use_cudnn_on_gpu,
                explicit_paddings,
                data_format,
                dilations
            });

            return _op.outputs[0];
        }

        public static Tensor bias_add(Tensor value,
            IVariableV1 bias,
            string data_format = null,
            string name = null)
        {
            if (tf.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "BiasAdd", name,
                    null,
                    value, bias,
                    "data_format", data_format);

                return results[0];
            }

            if (data_format == null)
                data_format = "NHWC";

            var _op = tf.OpDefLib._apply_op_helper("BiasAdd", name: name, args: new
            {
                value,
                bias,
                data_format
            });

            return _op.outputs[0];
        }

        public static Tensor bias_add_grad(Tensor out_backprop,
            string data_format = "NHWC",
            string name = null)
        {
            if (tf.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "BiasAddGrad", name,
                    null,
                    out_backprop,
                    "data_format", data_format);

                return results[0];
            }

            if (data_format == null)
                data_format = "NHWC";

            var _op = tf.OpDefLib._apply_op_helper("BiasAddGrad", name: name, args: new
            {
                out_backprop,
                data_format
            });

            return _op.outputs[0];
        }

        /// <summary>
        /// Computes exponential linear: <c>exp(features) - 1</c> if &amp;lt; 0, <c>features</c> otherwise.
        /// </summary>
        /// <param name="features">
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Elu'.
        /// </param>
        /// <returns>
        ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        /// <remarks>
        ///    See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
        ///    ](http://arxiv.org/abs/1511.07289)
        /// </remarks>
        public static Tensor elu(Tensor features, string name = "Elu")
        {
            var op = tf.OpDefLib._apply_op_helper("Elu", name: name, args: new { features });
            return op.output;
        }

        /// <summary>
        /// Gradient for batch normalization.
        /// </summary>
        /// <param name="params"></param>
        /// <returns></returns>
        public static Tensor[] fused_batch_norm_grad(FusedBatchNormParams @params)
        {
            var op = tf.OpDefLib._apply_op_helper("FusedBatchNormGrad", name: @params.Name, args: new
            {
                y_backprop = @params.YBackprop,
                x = @params.X,
                scale = @params.Scale,
                reserve_space_1 = @params.ReserveSpace1,
                reserve_space_2 = @params.ReserveSpace2,
                epsilon = @params.Epsilon,
                data_format = @params.DataFormat,
                is_training = @params.IsTraining
            });
            return op.outputs;
        }

        public static Tensor[] fused_batch_norm_grad_v3(FusedBatchNormParams @params)
            => tf.Context.RunInAutoMode(()
                => tf.OpDefLib._apply_op_helper("FusedBatchNormGradV3", name: @params.Name,
                    args: new 
                    {
                        y_backprop = @params.YBackprop,
                        x = @params.X,
                        scale = @params.Scale,
                        reserve_space_1 = @params.ReserveSpace1,
                        reserve_space_2 = @params.ReserveSpace2,
                        reserve_space_3 = @params.ReserveSpace3,
                        epsilon = @params.Epsilon,
                        data_format = @params.DataFormat,
                        is_training = @params.IsTraining
                    }).outputs, ()
                => tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "FusedBatchNormGradV3", @params.Name,
                    null,
                    @params.YBackprop, @params.X, @params.Scale,
                    @params.ReserveSpace1, @params.ReserveSpace2, @params.ReserveSpace3,
                    "epsilon", @params.Epsilon, 
                    "data_format", @params.DataFormat, 
                    "is_training", @params.IsTraining),
                @params.YBackprop);

        public static Tensor[] fused_batch_norm(Tensor x,
                Tensor scale,
                Tensor offset,
                Tensor mean,
                Tensor variance,
                float epsilon = 0.0001f,
                string data_format = "NHWC",
                bool is_training = true,
                string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("FusedBatchNorm", name: name, args: new
            {
                x,
                scale,
                offset,
                mean,
                variance,
                epsilon,
                data_format,
                is_training
            });

            return _op.outputs;
        }

        public static Tensors fused_batch_norm_v3(Tensor x,
            IVariableV1 scale,
            IVariableV1 offset,
            IVariableV1 mean,
            IVariableV1 variance,
            float epsilon = 0.0001f,
            float exponential_avg_factor = 1.0f,
            string data_format = "NHWC",
            bool is_training = true,
            string name = null)
        {
            if (tf.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "FusedBatchNormV3", name,
                    null,
                    x,
                    scale,
                    offset,
                    mean,
                    variance,
                    "epsilon", epsilon,
                    "exponential_avg_factor", exponential_avg_factor,
                    "data_format", data_format,
                    "is_training", is_training);

                return results;
            }

            var _op = tf.OpDefLib._apply_op_helper("FusedBatchNormV3", name: name, args: new
            {
                x,
                scale,
                offset,
                mean,
                variance,
                epsilon,
                data_format,
                is_training
            });

            return _op.outputs;
        }

        /// <summary>
        /// Local Response Normalization.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="depth_radius"></param>
        /// <param name="bias"></param>
        /// <param name="alpha"></param>
        /// <param name="beta"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor local_response_normalization(Tensor input, int depth_radius = 5, int bias = 1,
            int alpha = 1, float beta = 0.5f, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("LRN", name: name, args: new
            {
                input,
                depth_radius,
                bias,
                alpha,
                beta
            });

            return _op.output;
        }

        public static Tensor log_softmax(Tensor logits, string name = null)
            => tf.Context.RunInAutoMode(()
                => tf.OpDefLib._apply_op_helper("LogSoftmax", name: name,
                    args: new { logits }).output, ()
                => tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "LogSoftmax", name,
                    null,
                    logits).FirstOrDefault(),
                logits);

        /// <summary>
        /// Says whether the targets are in the top `K` predictions.
        /// </summary>
        /// <param name="predictions"></param>
        /// <param name="targets"></param>
        /// <param name="k"></param>
        /// <param name="name"></param>
        /// <returns>A `Tensor` of type `bool`.</returns>
        public static Tensor in_top_kv2(Tensor predictions, Tensor targets, int k, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("InTopKV2", name: name, args: new
            {
                predictions,
                targets,
                k
            });

            return _op.output;
        }

        public static Tensor leaky_relu(Tensor features, float alpha = 0.2f, string name = null)
            => tf.Context.RunInAutoMode(()
                => tf.OpDefLib._apply_op_helper("LeakyRelu", name: name,
                    args: new
                    {
                        features,
                        alpha
                    }).output, ()
                => tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "LeakyRelu", name,
                    null,
                    features,
                    "alpha", alpha).FirstOrDefault(),
                features);

        public static Tensor max_pool(Tensor input,
            int[] ksize,
            int[] strides,
            string padding,
            string data_format = "NHWC",
            string name = null)
        {
            if (tf.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "MaxPool", name,
                    null,
                    input,
                    "ksize", ksize,
                    "strides", strides,
                    "padding", padding,
                    "data_format", data_format);

                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("MaxPool", name: name, args: new
            {
                input,
                ksize,
                strides,
                padding,
                data_format,
            });

            return _op.outputs[0];
        }

        public static Tensor max_pool_grad(Tensor orig_input, Tensor orig_output, Tensor grad, int[] ksize, int[] strides, string padding,
            string data_format = "NHWC", string name = null)
        {
            if (tf.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "MaxPoolGrad", name,
                    null,
                    orig_input, orig_output, grad,
                    "ksize", ksize,
                    "strides", strides,
                    "padding", padding,
                    "data_format", data_format);

                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("MaxPoolGrad", name: name, args: new
            {
                orig_input,
                orig_output,
                grad,
                ksize,
                strides,
                padding,
                data_format
            });

            return _op.outputs[0];
        }

        public static Tensor[] top_kv2(Tensor input, int k, bool sorted = true, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("TopKV2", name: name, args: new
            {
                input,
                k,
                sorted
            });

            return _op.outputs;
        }

        public static Tensor relu_grad(Tensor gradients, Tensor features, string name = null)
        {
            if (tf.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "ReluGrad", name,
                    null,
                    gradients, features);

                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("ReluGrad", name: name, args: new
            {
                gradients,
                features
            });

            return _op.outputs[0];
        }

        public static Tensor leaky_relu_grad(Tensor gradients, Tensor features, float alpha = 0.2f, string name = null)
        {
            if (tf.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "LeakyReluGrad", name,
                    null,
                    gradients, features,
                    "alpha", alpha);

                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("LeakyReluGrad", name: name, args: new
            {
                gradients,
                features,
                alpha
            });

            return _op.output;
        }

        public static Tensor softmax(Tensor logits, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "Softmax", name,
                    null,
                    logits);

                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("Softmax", name: name, args: new
            {
                logits
            });

            return _op.outputs[0];
        }

        /// <summary>
        /// Computes softmax cross entropy cost and gradients to backpropagate.
        /// </summary>
        /// <param name="features"></param>
        /// <param name="labels"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static (Tensor, Tensor) softmax_cross_entropy_with_logits(Tensor features, Tensor labels, string name = null)
        {
            if (tf.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "SoftmaxCrossEntropyWithLogits", name,
                    null,
                    features, labels);

                return (results[0], results[1]);
            }

            var _op = tf.OpDefLib._apply_op_helper("SoftmaxCrossEntropyWithLogits", name: name, args: new
            {
                features,
                labels
            });

            return (_op.outputs[0], _op.outputs[1]);
        }

        /// <summary>
        ///    Computes softmax cross entropy cost and gradients to backpropagate.
        /// </summary>
        /// <param name="features">
        ///    batch_size x num_classes matrix
        /// </param>
        /// <param name="labels">
        ///    batch_size vector with values in [0, num_classes).
        ///    This is the label for the given minibatch entry.
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'SparseSoftmaxCrossEntropyWithLogits'.
        /// </param>
        /// <returns>
        ///    Returns a tuple with multiple values, as follows:
        ///    loss : Per example loss (batch_size vector).
        ///    backprop : backpropagated gradients (batch_size x num_classes matrix).
        ///    The Operation can be fetched from any of the Tensorreturned in the tuple values, by fetching the Operation property.
        /// </returns>
        /// <remarks>
        ///    Unlike <c>SoftmaxCrossEntropyWithLogits</c>, this operation does not accept
        ///    a matrix of label probabilities, but rather a single label per row
        ///    of features.  This label is considered to have probability 1.0 for the
        ///    given row.
        ///    
        ///    Inputs are the logits, not probabilities.
        /// </remarks>
        public static (Tensor loss, Tensor backprop) sparse_softmax_cross_entropy_with_logits(Tensor features, Tensor labels, string name = "SparseSoftmaxCrossEntropyWithLogits")
        {
            if (tf.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "SparseSoftmaxCrossEntropyWithLogits", name,
                    null,
                    features, labels);

                return (results[0], results[1]);
            }

            var op = tf.OpDefLib._apply_op_helper("SparseSoftmaxCrossEntropyWithLogits", name: name, args: new { features, labels });
            int _idx = 0;
            var loss = op.outputs[_idx++];
            var backprop = op.outputs[_idx++];
            return (loss, backprop);
        }

        /// <summary>
        /// Computes rectified linear: `max(features, 0)`.
        /// </summary>
        /// <param name="features">A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`, `qint8`.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A `Tensor`. Has the same type as `features`.</returns>
        public static Tensor relu(Tensor features, string name = null)
        {
            if (tf.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "Relu", name,
                    null,
                    features);

                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("Relu", name: name, args: new { features });
            return _op.outputs[0];
        }

        public static Tensor tanh(Tensor x, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "Tanh", name,
                    null,
                    x);

                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("Tanh", name: name, args: new { x });
            return _op.outputs[0];
        }
    }
}
