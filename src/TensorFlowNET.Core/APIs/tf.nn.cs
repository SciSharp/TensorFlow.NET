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

using Tensorflow.Operations;
using Tensorflow.Operations.Activation;
using static Tensorflow.Python;

namespace Tensorflow
{
    public static partial class tf
    {
        public static class nn
        {
            public static Tensor conv2d(Tensor input, RefVariable filter, int[] strides, string padding, bool use_cudnn_on_gpu = true, 
                string data_format= "NHWC", int[] dilations= null, string name = null)
            {
                var parameters = new Conv2dParams
                {
                    Input = input,
                    Filter = filter,
                    Strides = strides,
                    Padding = padding,
                    UseCudnnOnGpu = use_cudnn_on_gpu,
                    DataFormat = data_format,
                    Name = name
                };

                if (dilations != null)
                    parameters.Dilations = dilations;

                return gen_nn_ops.conv2d(parameters);
            }

            /// <summary>
            /// Computes dropout.
            /// </summary>
            /// <param name="x">A floating point tensor.</param>
            /// <param name="keep_prob">(deprecated) A deprecated alias for `(1-rate)`.</param>
            /// <param name="noise_shape"></param>
            /// <param name="seed">Used to create random seeds.</param>
            /// <param name="name"></param>
            /// <param name="rate">A scalar `Tensor` with the same type as `x`.</param>
            /// <returns>A Tensor of the same shape of `x`.</returns>
            public static Tensor dropout(Tensor x, Tensor keep_prob = null, Tensor noise_shape = null, int? seed = null, string name = null,
                float? rate = null)
            {
                Tensor keep = null;
                if (keep_prob != null)
                    keep = 1.0f - keep_prob;
                var rate_tensor = keep;

                return nn_ops.dropout_v2(x, rate: rate_tensor, noise_shape: noise_shape, seed: seed, name: name);
            }

            /// <summary>
            /// Creates a recurrent neural network specified by RNNCell `cell`.
            /// </summary>
            /// <param name="cell">An instance of RNNCell.</param>
            /// <param name="inputs">The RNN inputs.</param>
            /// <param name="dtype"></param>
            /// <param name="swap_memory"></param>
            /// <param name="time_major"></param>
            /// <returns>A pair (outputs, state)</returns>
            public static (Tensor, Tensor) dynamic_rnn(RNNCell cell, Tensor inputs,
                Tensor sequence_length = null, TF_DataType dtype = TF_DataType.DtInvalid,
                int? parallel_iterations = null, bool swap_memory = false, bool time_major = false)
                => rnn.dynamic_rnn(cell, inputs, sequence_length: sequence_length, dtype: dtype,
                    parallel_iterations: parallel_iterations, swap_memory: swap_memory,
                    time_major: time_major);

            public static Tensor elu(Tensor features, string name = null)
                => gen_nn_ops.elu(features, name: name);

            public static (Tensor, Tensor) moments(Tensor x,
                int[] axes,
                string name = null,
                bool keep_dims = false) => nn_impl.moments(x, 
                    axes, 
                    name: name, 
                    keep_dims: keep_dims);

            public static Tensor embedding_lookup(RefVariable @params,
                Tensor ids,
                string partition_strategy = "mod",
                string name = null) => embedding_ops._embedding_lookup_and_transform(@params,
                    ids,
                    partition_strategy: partition_strategy,
                    name: name);

            public static Tensor embedding_lookup(Tensor @params,
                Tensor ids,
                string partition_strategy = "mod",
                string name = null) => embedding_ops._embedding_lookup_and_transform(new Tensor[] { @params },
                    ids,
                    partition_strategy: partition_strategy,
                    name: name);

            public static IActivation relu() => new relu();

            public static Tensor relu(Tensor features, string name = null) => gen_nn_ops.relu(features, name);

            public static Tensor[] fused_batch_norm(Tensor x,
                RefVariable scale,
                RefVariable offset,
                Tensor mean = null,
                Tensor variance = null,
                float epsilon = 0.001f,
                string data_format = "NHWC",
                bool is_training = true,
                string name = null) => nn_impl.fused_batch_norm(x, scale, offset, mean, variance,
                    epsilon: epsilon,
                    data_format: data_format,
                    is_training: is_training,
                    name: name);

            public static IPoolFunction max_pool_fn => new MaxPoolFunction();

            public static Tensor max_pool(Tensor value, int[] ksize, int[] strides, string padding, string data_format = "NHWC", string name = null) 
                => nn_ops.max_pool(value, ksize, strides, padding, data_format: data_format, name: name);

            public static Tensor in_top_k(Tensor predictions, Tensor targets, int k, string name = "InTopK") => gen_ops.in_top_k(predictions, targets, k, name);

            public static Tensor[] top_k(Tensor input, int k = 1, bool sorted = true, string name = null)
                => gen_nn_ops.top_kv2(input, k: k, sorted: sorted, name: name);

            public static Tensor bias_add(Tensor value, RefVariable bias, string data_format = null, string name = null)
            {
                return Python.tf_with(ops.name_scope(name, "BiasAdd", new { value, bias }), scope =>
                {
                    name = scope;
                    return gen_nn_ops.bias_add(value, bias, data_format: data_format, name: name);
                });
            }

            public static rnn_cell_impl rnn_cell => new rnn_cell_impl();

            public static Tensor softmax(Tensor logits, int axis = -1, string name = null)
                => gen_nn_ops.softmax(logits, name);

            /// <summary>
            /// Computes sparse softmax cross entropy between `logits` and `labels`.
            /// </summary>
            /// <param name="labels"></param>
            /// <param name="logits"></param>
            /// <param name="name"></param>
            /// <returns></returns>
            public static Tensor sparse_softmax_cross_entropy_with_logits(Tensor labels = null,
            Tensor logits = null, string name = null)
                => nn_ops.sparse_softmax_cross_entropy_with_logits(labels: labels, logits: logits, name: name);

            /// <summary>
            /// Computes softmax cross entropy between `logits` and `labels`.
            /// </summary>
            /// <param name="labels"></param>
            /// <param name="logits"></param>
            /// <param name="dim"></param>
            /// <param name="name"></param>
            /// <returns></returns>
            public static Tensor softmax_cross_entropy_with_logits(Tensor labels, Tensor logits, int dim = -1, string name = null)
            {
                tf_with(ops.name_scope(name, "softmax_cross_entropy_with_logits_sg", new { logits, labels }), scope =>
                {
                    name = scope;
                    labels = array_ops.stop_gradient(labels, name: "labels_stop_gradient");
                });

                return softmax_cross_entropy_with_logits_v2(labels, logits, axis: dim, name: name);
            }

            public static Tensor softmax_cross_entropy_with_logits_v2(Tensor labels, Tensor logits, int axis = -1, string name = null)
                => nn_ops.softmax_cross_entropy_with_logits_v2_helper(labels, logits, axis: axis, name: name);
        }
    }
}
