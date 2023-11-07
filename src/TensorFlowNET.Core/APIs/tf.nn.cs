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

using System.Xml.Linq;
using Tensorflow.Operations;
using Tensorflow.Operations.Activation;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public nn_internal nn { get; } = new nn_internal();

        public class nn_internal
        {
            public Tensor conv2d(Tensor input, Tensor filter, int[] strides, string padding, bool use_cudnn_on_gpu = true,
                string data_format = "NHWC", int[] dilations = null, string name = null)
            {
                return gen_nn_ops.conv2d(input, filter, strides, padding, use_cudnn_on_gpu, 
                    data_format: data_format, dilations: dilations, name: name);
            }

            public Tensor[] ctc_greedy_decoder(Tensor inputs, Tensor sequence_length, bool merge_repeated = true, string name = null)
                => gen_ctc_ops.ctc_greedy_decoder(inputs, sequence_length, merge_repeated: merge_repeated, name: name);

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
            public Tensor dropout(Tensor x, Tensor keep_prob = null, Tensor noise_shape = null, int? seed = null, string name = null,
                float? rate = null)
            {
                Tensor keep = null;
                if (keep_prob != null)
                    keep = 1.0f - keep_prob;
                var rate_tensor = rate.HasValue ? tf.constant(rate.Value) : keep;
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
            public (Tensor, Tensor) dynamic_rnn(RnnCell cell, Tensor inputs,
                Tensor sequence_length = null, TF_DataType dtype = TF_DataType.DtInvalid,
                int? parallel_iterations = null, bool swap_memory = false, bool time_major = false)
                => rnn.dynamic_rnn(cell, inputs, sequence_length: sequence_length, dtype: dtype,
                    parallel_iterations: parallel_iterations, swap_memory: swap_memory,
                    time_major: time_major);

            public Tensor elu(Tensor features, string name = null)
                => gen_nn_ops.elu(features, name: name);

            public (Tensor, Tensor) moments(Tensor x,
                Axis axes,
                string name = null,
                bool keep_dims = false) => nn_impl.moments(x,
                    axes,
                    name: name,
                    keep_dims: keep_dims);

            public Tensor embedding_lookup(IVariableV1 @params,
                Tensor ids,
                string partition_strategy = "mod",
                string name = null) => embedding_ops._embedding_lookup_and_transform(@params,
                    ids,
                    partition_strategy: partition_strategy,
                    name: name);

            public Tensor embedding_lookup(Tensor @params,
                Tensor ids,
                string partition_strategy = "mod",
                string name = null) => embedding_ops._embedding_lookup_and_transform(new Tensor[] { @params },
                    ids,
                    partition_strategy: partition_strategy,
                    name: name);

            public IActivation relu() => new relu();


            public IActivation swish() => new swish();
            public IActivation tanh() => new tanh();

            public IActivation softmax() => new softmax();
            public Tensor tanh(Tensor x, string name = null)
                => gen_math_ops.tanh(x, name);

            public Tensor relu(Tensor features, string name = null)
                => gen_nn_ops.relu(features, name);

            public Tensor relu6(Tensor features, string name = null)
                    => gen_nn_ops.relu6(features, name);

            public Tensor[] fused_batch_norm(Tensor x,
                Tensor scale,
                Tensor offset,
                Tensor mean = null,
                Tensor variance = null,
                float epsilon = 0.001f,
                string data_format = "NHWC",
                bool is_training = true,
                string name = null,
                float exponential_avg_factor = 1.0f) => nn_impl.fused_batch_norm(x, scale, offset, mean, variance,
                    epsilon: epsilon,
                    data_format: data_format,
                    is_training: is_training,
                    name: name,
                    exponential_avg_factor: exponential_avg_factor);

            /// <summary>
            /// Normalizes a tensor by `mean` and `variance`, and applies (optionally) a`scale` \\(\gamma\\) to it, as well as an `offset` \\(\beta\\).
            /// </summary>
            /// <param name="x">A floating point tensor.</param>
            /// <param name="mean">A mean `Tensor`.</param>
            /// <param name="variance">A variance `Tensor`.</param>
            /// <param name="offset"> An offset `Tensor`, often denoted \\(\beta\\) in equations, or NULL. If present, will be added to the normalized tensor.</param>
            /// <param name="scale"> A scale `Tensor`, often denoted \\(\gamma\\) in equations, or NULL. If present, the scale is applied to the normalized tensor.</param>
            /// <param name="variance_epsilon"> A small float number to avoid dividing by 0.</param>
            /// <param name="name">A name for this operation.</param>
            /// <returns>the normalized, scaled, offset tensor.</returns>
            public Tensor batch_normalization(Tensor x,
                        Tensor mean,
                        Tensor variance,
                        Tensor offset,
                        Tensor scale,
                        float variance_epsilon,
                        string name = null) => nn_impl.batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name);


            public Tensor max_pool(Tensor value, int[] ksize, int[] strides, string padding, string data_format = "NHWC", string name = null)
                => nn_ops.max_pool(value, ksize, strides, padding, data_format: data_format, name: name);

            public Tensor in_top_k(Tensor predictions, Tensor targets, int k, string name = "InTopK")
                => nn_ops.in_top_k(predictions, targets, k, name);

            public Tensor[] top_k(Tensor input, int k = 1, bool sorted = true, string name = null)
                => gen_nn_ops.top_kv2(input, k: ops.convert_to_tensor(k), sorted: sorted, name: name);

            public Tensor bias_add(Tensor value, IVariableV1 bias, string data_format = null, string name = null)
            {
                return tf_with(ops.name_scope(name, "BiasAdd", new { value, bias }), scope =>
                {
                    name = scope;
                    return gen_nn_ops.bias_add(value, ops.convert_to_tensor(bias), data_format: data_format, name: name);
                });
            }

            public Tensor l2_loss(Tensor t, string name = null)
                => nn_ops.l2_loss(t, name: name);

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
            public Tensor lrn(Tensor input, int depth_radius = 5, int bias = 1,
                int alpha = 1, float beta = 0.5f, string name = null)
                => gen_nn_ops.lrn(input, depth_radius: depth_radius, bias: bias,
                    alpha: alpha, beta: beta, name: name);

            public Tensor leaky_relu(Tensor features, float alpha = 0.2f, string name = null)
                => nn_ops.leaky_relu(features, alpha: alpha, name: name);

            public rnn_cell_impl rnn_cell => new rnn_cell_impl();

            public Tensor sigmoid_cross_entropy_with_logits(Tensor labels, Tensor logits, string name = null)
                => nn_impl.sigmoid_cross_entropy_with_logits(labels: labels, logits: logits, name: name);

            public Tensor softmax(Tensor logits, int axis = -1, string name = null)
                => gen_nn_ops.softmax(logits, name);


            /// <summary>
            /// Computes sparse softmax cross entropy between `logits` and `labels`.
            /// </summary>
            /// <param name="labels"></param>
            /// <param name="logits"></param>
            /// <param name="name"></param>
            /// <returns></returns>
            public Tensor sparse_softmax_cross_entropy_with_logits(Tensor labels = null,
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
            public Tensor softmax_cross_entropy_with_logits(Tensor labels, Tensor logits, int dim = -1, string name = null)
            {
                tf_with(ops.name_scope(name, "softmax_cross_entropy_with_logits_sg", new { logits, labels }), scope =>
                {
                    name = scope;
                    labels = array_ops.stop_gradient(labels, name: "labels_stop_gradient");
                });

                return softmax_cross_entropy_with_logits_v2(labels, logits, axis: dim, name: name);
            }

            public Tensor softmax_cross_entropy_with_logits_v2(Tensor labels, Tensor logits, int axis = -1, string name = null)
                => nn_ops.softmax_cross_entropy_with_logits_v2_helper(labels, logits, axis: axis, name: name);

            /// <summary>
            /// Computes sigmoid of `x` element-wise.
            /// Specifically, `y = 1 / (1 + exp(-x))`.
            /// </summary>
            /// <typeparam name="T"></typeparam>
            /// <param name="x"></param>
            /// <param name="name">A name for the operation (optional).</param>
            /// <returns>A Tensor with the same type as `x`.</returns>
            public Tensor sigmoid<T>(T x, string name = null)
                => math_ops.sigmoid(x, name: name);
        }
    }
}
