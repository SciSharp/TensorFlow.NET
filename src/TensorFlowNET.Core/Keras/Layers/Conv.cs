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
using System.Collections.Generic;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;
using Tensorflow.Operations;
using Tensorflow.Operations.Activation;

namespace Tensorflow.Keras.Layers
{
    public class Conv : Tensorflow.Layers.Layer
    {
        protected int rank;
        protected int filters;
        protected int[] kernel_size;
        protected int[] strides;
        protected string padding;
        protected string data_format;
        protected int[] dilation_rate;
        protected IActivation activation;
        protected bool use_bias;
        protected IInitializer kernel_initializer;
        protected IInitializer bias_initializer;
        protected RefVariable kernel;
        protected RefVariable bias;
        protected Convolution _convolution_op;

        public Conv(int rank, 
            int filters,
            int[] kernel_size,
            int[] strides = null,
            string padding = "valid",
            string data_format = null,
            int[] dilation_rate = null,
            IActivation activation = null,
            bool use_bias = true,
            IInitializer kernel_initializer = null,
            IInitializer bias_initializer = null,
            bool trainable = true, 
            string name = null) : base(trainable: trainable, name: name)
        {
            this.rank = rank;
            this.filters = filters;
            this.kernel_size = kernel_size;
            this.strides = strides;
            this.padding = padding;
            this.data_format = data_format;
            this.dilation_rate = dilation_rate;
            this.activation = activation;
            this.use_bias = use_bias;
            this.kernel_initializer = kernel_initializer;
            this.bias_initializer = bias_initializer;
            inputSpec = new InputSpec(ndim: rank + 2);
        }

        protected override void build(TensorShape input_shape)
        {
            int channel_axis = data_format == "channels_first" ? 1 : -1;
            int input_dim = channel_axis < 0 ? 
                input_shape.dims[input_shape.ndim + channel_axis] : 
                input_shape.dims[channel_axis];
            var kernel_shape = new int[] { kernel_size[0], kernel_size[1], input_dim, filters };
            kernel = (RefVariable)add_weight(name: "kernel",
                shape: kernel_shape,
                initializer: kernel_initializer,
                trainable: true,
                dtype: DType);
            if (use_bias)
                bias = (RefVariable)add_weight(name: "bias",
                    shape: new int[] { filters },
                    initializer: bias_initializer,
                    trainable: true,
                    dtype: DType);

            var axes = new Dictionary<int, int>();
            axes.Add(-1, input_dim);
            inputSpec = new InputSpec(ndim: rank + 2, axes: axes);

            string op_padding;
            if (padding == "causal")
                op_padding = "valid";
            else
                op_padding = padding;

            var df = conv_utils.convert_data_format(data_format, rank + 2);
            _convolution_op = nn_ops.Convolution(input_shape,
                kernel.shape,
                op_padding.ToUpper(),
                strides,
                dilation_rate,
                data_format: df);

            built = true;
        }

        protected override Tensor call(Tensor inputs, bool training = false, Tensor state = null)
        {
            var outputs = _convolution_op.__call__(inputs, kernel);
            if (use_bias)
            {
                if (data_format == "channels_first")
                {
                    throw new NotImplementedException("call channels_first");
                }
                else
                {                    
                    outputs = nn_ops.bias_add(outputs, bias, data_format: "NHWC");
                }
            }

            if (activation != null)
                outputs = activation.Activate(outputs);

            return outputs;
        }
    }
}
