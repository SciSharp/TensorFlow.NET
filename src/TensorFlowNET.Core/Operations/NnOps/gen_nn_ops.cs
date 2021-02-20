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
            => tf.Context.ExecuteOp("Conv2D", parameters.Name, new ExecuteOpArgs(parameters.Input, parameters.Filter)
                .SetAttributes(new
                {
                    strides = parameters.Strides,
                    padding = parameters.Padding,
                    use_cudnn_on_gpu = parameters.UseCudnnOnGpu,
                    explicit_paddings = parameters.ExplicitPaddings,
                    data_format = parameters.DataFormat,
                    dilations = parameters.Dilations
                }));

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
                => tf.Context.ExecuteOp("Conv2DBackpropFilter", name, new ExecuteOpArgs(input, filter_sizes, out_backprop)
                    .SetAttributes(new
                    {
                        strides,
                        padding,
                        use_cudnn_on_gpu,
                        explicit_paddings = explicit_paddings ?? new int[0],
                        data_format,
                        dilations = dilations ?? new int[] { 1, 1, 1, 1 }
                    }));

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
                => tf.Context.ExecuteOp("Conv2DBackpropInput", name, new ExecuteOpArgs(input_sizes, filter, out_backprop)
                    .SetAttributes(new
                    {
                        strides,
                        padding,
                        use_cudnn_on_gpu,
                        explicit_paddings = explicit_paddings ?? new int[0],
                        data_format,
                        dilations = dilations ?? new int[] { 1, 1, 1, 1 }
                    }));

        public static Tensor bias_add(Tensor value,
            IVariableV1 bias,
            string data_format = null,
            string name = null)
                => tf.Context.ExecuteOp("BiasAdd", name, new ExecuteOpArgs(value, bias)
                    .SetAttributes(new { data_format = data_format ?? "NHWC" }));

        public static Tensor bias_add_grad(Tensor out_backprop,
            string data_format = "NHWC",
            string name = null)
                => tf.Context.ExecuteOp("BiasAddGrad", name, new ExecuteOpArgs(out_backprop)
                    .SetAttributes(new { data_format = data_format ?? "NHWC" }));

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
            => tf.Context.ExecuteOp("FusedBatchNormGradV3", @params.Name,
                new ExecuteOpArgs(@params.YBackprop,
                    @params.X,
                    @params.Scale,
                    @params.ReserveSpace1,
                    @params.ReserveSpace2,
                    @params.ReserveSpace3)
                .SetAttributes(new
                {
                    epsilon = @params.Epsilon,
                    data_format = @params.DataFormat,
                    is_training = @params.IsTraining
                }));

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
                => tf.Context.ExecuteOp("FusedBatchNormV3", name, new ExecuteOpArgs(x, scale, offset, mean, variance)
                    .SetAttributes(new { epsilon, data_format, is_training }));

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
            => tf.Context.ExecuteOp("LogSoftmax", name, new ExecuteOpArgs(logits));

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
            => tf.Context.ExecuteOp("LeakyRelu", name,
                new ExecuteOpArgs(features).SetAttributes(new { alpha }));

        public static Tensor max_pool(Tensor input,
            int[] ksize,
            int[] strides,
            string padding,
            string data_format = "NHWC",
            string name = null)
                => tf.Context.ExecuteOp("MaxPool", name, new ExecuteOpArgs(input)
                    .SetAttributes(new
                    {
                        ksize,
                        strides,
                        padding,
                        data_format
                    }));

        public static Tensor max_pool_grad(Tensor orig_input, Tensor orig_output, Tensor grad, int[] ksize, int[] strides, string padding,
            string data_format = "NHWC", string name = null)
                => tf.Context.ExecuteOp("MaxPoolGrad", name, new ExecuteOpArgs(orig_input, orig_output, grad)
                    .SetAttributes(new
                    {
                        ksize,
                        strides,
                        padding,
                        data_format
                    }));

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
            => tf.Context.ExecuteOp("ReluGrad", name, new ExecuteOpArgs(gradients, features));

        public static Tensor leaky_relu_grad(Tensor gradients, Tensor features, float alpha = 0.2f, string name = null)
            => tf.Context.ExecuteOp("LeakyReluGrad", name, new ExecuteOpArgs(gradients, features)
                .SetAttributes(new { alpha }));

        public static Tensor softmax(Tensor logits, string name = null)
            => tf.Context.ExecuteOp("Softmax", name, new ExecuteOpArgs(logits));

        /// <summary>
        /// Computes softmax cross entropy cost and gradients to backpropagate.
        /// </summary>
        /// <param name="features"></param>
        /// <param name="labels"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static (Tensor, Tensor) softmax_cross_entropy_with_logits(Tensor features, Tensor labels, string name = null)
        {
            var results = tf.Context.ExecuteOp("SoftmaxCrossEntropyWithLogits", name, new ExecuteOpArgs(features, labels));

            return (results[0], results[1]);
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
            var results = tf.Context.ExecuteOp("SparseSoftmaxCrossEntropyWithLogits", name, new ExecuteOpArgs(features, labels));

            return (results[0], results[1]);
        }

        /// <summary>
        /// Computes rectified linear: `max(features, 0)`.
        /// </summary>
        /// <param name="features">A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`, `qint8`.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A `Tensor`. Has the same type as `features`.</returns>
        public static Tensor relu(Tensor features, string name = null)
            => tf.Context.ExecuteOp("Relu", name, new ExecuteOpArgs(features));

        public static Tensor tanh(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Tanh", name, new ExecuteOpArgs(x));
    }
}
