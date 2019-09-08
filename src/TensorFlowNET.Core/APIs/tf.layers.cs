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

using System.Collections.Generic;
using System.Linq;
using NumSharp;
using Tensorflow.Keras.Layers;
using Tensorflow.Operations.Activation;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public layers_internal layers { get; } = new layers_internal();

        public class layers_internal
        {
            public Tensor conv2d(Tensor inputs,
                int filters,
                int[] kernel_size,
                int[] strides = null,
                string padding = "valid",
                string data_format= "channels_last",
                int[] dilation_rate = null,
                bool use_bias = true,
                IActivation activation = null,
                IInitializer kernel_initializer = null,
                IInitializer bias_initializer = null,
                bool trainable = true,
                string name = null)
            {
                if (strides == null)
                    strides = new int[] { 1, 1 };
                if (dilation_rate == null)
                    dilation_rate = new int[] { 1, 1 };
                if (bias_initializer == null)
                    bias_initializer = tf.zeros_initializer;

                var layer = new Conv2D(filters,
                    kernel_size: kernel_size,
                    strides: strides,
                    padding: padding,
                    data_format: data_format,
                    dilation_rate: dilation_rate,
                    activation: activation,
                    use_bias: use_bias,
                    kernel_initializer: kernel_initializer,
                    bias_initializer: bias_initializer,
                    trainable: trainable,
                    name: name);

                return layer.apply(inputs);
            }

            /// <summary>
            /// Functional interface for the batch normalization layer.
            /// http://arxiv.org/abs/1502.03167
            /// </summary>
            /// <param name="inputs"></param>
            /// <param name="axis"></param>
            /// <param name="momentum"></param>
            /// <param name="epsilon"></param>
            /// <param name="center"></param>
            /// <param name="scale"></param>
            /// <param name="beta_initializer"></param>
            /// <param name="gamma_initializer"></param>
            /// <param name="moving_mean_initializer"></param>
            /// <param name="moving_variance_initializer"></param>
            /// <param name="training"></param>
            /// <param name="trainable"></param>
            /// <param name="name"></param>
            /// <param name="renorm"></param>
            /// <param name="renorm_momentum"></param>
            /// <returns></returns>
            public Tensor batch_normalization(Tensor inputs,
                int axis = -1,
                float momentum = 0.99f,
                float epsilon = 0.001f,
                bool center = true,
                bool scale = true,
                IInitializer beta_initializer = null,
                IInitializer gamma_initializer = null,
                IInitializer moving_mean_initializer = null,
                IInitializer moving_variance_initializer = null,
                Tensor training = null,
                bool trainable = true,
                string name = null,
                bool renorm = false,
                float renorm_momentum = 0.99f)
            {
                var layer = new BatchNormalization(
                    axis: axis,
                    momentum: momentum,
                    epsilon: epsilon,
                    center: center,
                    scale: scale,
                    beta_initializer: beta_initializer,
                    gamma_initializer: gamma_initializer,
                    moving_mean_initializer: moving_mean_initializer,
                    moving_variance_initializer: moving_variance_initializer,
                    renorm: renorm,
                    renorm_momentum: renorm_momentum,
                    trainable: trainable,
                    name: name);

                return layer.apply(inputs, training: training);
            }

            /// <summary>
            /// Max pooling layer for 2D inputs (e.g. images).
            /// </summary>
            /// <param name="inputs">The tensor over which to pool. Must have rank 4.</param>
            /// <param name="pool_size"></param>
            /// <param name="strides"></param>
            /// <param name="padding"></param>
            /// <param name="data_format"></param>
            /// <param name="name"></param>
            /// <returns></returns>
            public Tensor max_pooling2d(Tensor inputs,
                int[] pool_size,
                int[] strides,
                string padding = "valid",
                string data_format = "channels_last",
                string name = null)
            {
                var layer = new MaxPooling2D(pool_size: pool_size,
                    strides: strides,
                    padding: padding,
                    data_format: data_format,
                    name: name);

                return layer.apply(inputs);
            }

            public Tensor dense(Tensor inputs,
                int units,
                IActivation activation = null,
                bool use_bias = true,
                IInitializer kernel_initializer = null,
                IInitializer bias_initializer = null,
                bool trainable = true,
                string name = null,
                bool? reuse = null)
            {
                if (bias_initializer == null)
                    bias_initializer = tf.zeros_initializer;

                var layer = new Dense(units, activation, 
                    use_bias: use_bias,
                    bias_initializer: bias_initializer,
                    kernel_initializer: kernel_initializer);

                return layer.apply(inputs);
            }

            /// <summary>
            ///     Flattens an input tensor while preserving the batch axis (axis 0).
            /// </summary>
            /// <param name="inputs">Tensor input.</param>
            /// <param name="name">The name of the layer.</param>
            /// <param name="data_format">
            ///     A string, one of `channels_last` (default) or `channels_first`. <br></br>
            ///     The ordering of the dimensions in the inputs. <br></br>
            ///     `channels_last` corresponds to inputs with shape <br></br>
            ///     `(batch, height, width, channels)` while `channels_first` corresponds to <br></br>
            ///     inputs with shape `(batch, channels, height, width)`. 
            /// </param>
            /// <returns></returns>
            public Tensor flatten(Tensor inputs,
                string name = null,
                string data_format = "channels_last")
            {
                var input_shape = inputs.shape;
                if (inputs.shape.Length == 0)
                    throw new ValueError($"Input 0 of layer flatten is incompatible with the layer: : expected min_ndim={1}, found ndim={0}. Full shape received: ()");

                var premutation = new List<int>() {0};
                if (data_format == "channels_first" && inputs.NDims > 1)
                {
                    premutation.AddRange(Binding.range(2, inputs.NDims));
                    premutation.Add(1);
                    inputs = array_ops.transpose(inputs, premutation.ToArray());
                }

                var ret = array_ops.reshape(inputs, compute_output_shape(input_shape));
                //ret.set_shape(compute_output_shape(ret.shape));
                return ret;

                int[] compute_output_shape(int[] inputshape)
                {
                    if (inputshape == null || inputshape.Length == 0)
                        inputshape = new int[] {1};

                    if (inputshape.Skip(1).All(d => d > 0))
                    {
                        int[] output_shape = new int[2];
                        output_shape[0] = inputshape[0];
                        output_shape[1] = inputshape.Skip(1).Aggregate(1, (acc, rhs) => acc*rhs); //calculate size of all the rest dimensions
                        return output_shape;
                    } else
                        return new int[] {inputshape[0], -1}; //-1 == Binding.None
                }
            }
        }
    }
}
