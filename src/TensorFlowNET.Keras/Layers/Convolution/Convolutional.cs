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
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;
using Tensorflow.Operations;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    public class Convolutional : Layer
    {
        ConvolutionalArgs args;
        protected int rank => args.Rank;
        protected int filters => args.Filters;
        protected TensorShape kernel_size => args.KernelSize;
        protected TensorShape strides => args.Strides;
        protected string padding => args.Padding;
        protected string data_format => args.DataFormat;
        protected TensorShape dilation_rate => args.DilationRate;
        protected Activation activation => args.Activation;
        protected bool use_bias => args.UseBias;
        protected IInitializer kernel_initializer => args.KernelInitializer;
        protected IRegularizer kernel_regularizer => args.KernelRegularizer;
        protected IInitializer bias_initializer => args.BiasInitializer;
        protected IVariableV1 kernel;
        protected IVariableV1 bias;
        ConvolutionInternal _convolution_op;
        protected string _tf_data_format;

        public Convolutional(ConvolutionalArgs args) : base(args)
        {
            this.args = args;
            args.KernelSize = conv_utils.normalize_tuple(args.KernelSize.dims, args.Rank, "kernel_size");
            args.Strides = conv_utils.normalize_tuple(args.Strides.dims, args.Rank, "strides");
            args.Padding = conv_utils.normalize_padding(args.Padding);
            args.DataFormat = conv_utils.normalize_data_format(args.DataFormat);
            args.DilationRate = conv_utils.normalize_tuple(args.DilationRate.dims, args.Rank, "dilation_rate");
            inputSpec = new InputSpec(ndim: rank + 2);
            _tf_data_format = conv_utils.convert_data_format(data_format, rank + 2);
        }

        protected override void build(Tensors inputs)
        {
            TensorShape input_shape = inputs.shape;
            int channel_axis = data_format == "channels_first" ? 1 : -1;
            int input_channel = channel_axis < 0 ?
                input_shape.dims[input_shape.ndim + channel_axis] :
                input_shape.dims[channel_axis];
            TensorShape kernel_shape = kernel_size.dims.concat(new int[] { input_channel / args.Groups, filters });
            kernel = add_weight(name: "kernel",
                shape: kernel_shape,
                initializer: kernel_initializer,
                regularizer: kernel_regularizer,
                trainable: true,
                dtype: DType);
            if (use_bias)
                bias = add_weight(name: "bias",
                    shape: new int[] { filters },
                    initializer: bias_initializer,
                    trainable: true,
                    dtype: DType);

            var axes = new Dictionary<int, int>();
            axes.Add(-1, input_channel);
            inputSpec = new InputSpec(min_ndim: rank + 2, axes: axes);

            string tf_padding;
            if (padding == "causal")
                tf_padding = "VALID";
            else
                tf_padding = padding.ToUpper();

            string tf_op_name = GetType().Name;


            _convolution_op = nn_ops.convolution_internal(tf_padding,
                strides,
                dilation_rate,
                rank,
                data_format: _tf_data_format,
                name: tf_op_name);

            built = true;
        }

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool? training = false)
        {
            var outputs = _convolution_op.Apply(inputs, kernel);
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
                outputs = activation(outputs);

            return outputs;
        }

        protected virtual int _get_channel_axis()
            => data_format == "channels_first" ? -1 - rank : -1;
    }
}
