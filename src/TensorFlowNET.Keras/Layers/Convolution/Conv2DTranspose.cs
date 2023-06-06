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
using static Tensorflow.Binding;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Utils;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Saving;
using Tensorflow.Common.Types;

namespace Tensorflow.Keras.Layers
{
    public class Conv2DTranspose : Conv2D
    {
        public Conv2DTranspose(Conv2DArgs args) : base(InitializeUndefinedArgs(args))
        {

        }

        private static Conv2DArgs InitializeUndefinedArgs(Conv2DArgs args)
        {
            if (args.Strides is null)
            {
                args.Strides = (1, 1);
            }
            if (string.IsNullOrEmpty(args.Padding))
            {
                args.Padding = "valid";
            }
            if (args.DilationRate == 0)
            {
                args.DilationRate = (1, 1);
            }
            if (args.Groups == 0)
            {
                args.Groups = 1;
            }
            if (args.KernelInitializer is null)
            {
                args.KernelInitializer = tf.glorot_uniform_initializer;
            }
            if (args.BiasInitializer is null)
            {
                args.BiasInitializer = tf.zeros_initializer;
            }
            return args;
        }

        public override void build(KerasShapesWrapper input_shape)
        {
            var single_shape = input_shape.ToSingleShape();
            if (len(single_shape) != 4)
                throw new ValueError($"Inputs should have rank 4. Received input shape: {input_shape}");

            var channel_axis = _get_channel_axis();
            var input_dim = single_shape[-1];
            var kernel_shape = new Shape(kernel_size[0], kernel_size[1], filters, input_dim);

            kernel = add_weight(name: "kernel",
                shape: kernel_shape,
                initializer: kernel_initializer,
                regularizer: kernel_regularizer,
                trainable: true);
            if (use_bias)
                bias = add_weight(name: "bias",
                shape: filters,
                initializer: bias_initializer,
                trainable: true);
            built = true;
            _buildInputShape = input_shape;
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var inputs_shape = array_ops.shape(inputs);
            var batch_size = inputs_shape[0];
            var (h_axis, w_axis) = (1, 2);
            if (data_format == "channels_first")
                (h_axis, w_axis) = (2, 3);
            var (height, width) = (-1, -1);
            if(inputs.shape.ndim > -1)
            {
                var dims = inputs.shape.dims;
                (height, width) = ((int)dims[h_axis], (int)dims[w_axis]);
            }
            var (kernel_h, kernel_w) = kernel_size;
            var (stride_h, stride_w) = strides;

            var (out_pad_h, out_pad_w) = (-1, -1);

            // Infer the dynamic output shape:
            var out_height = conv_utils.deconv_output_length(height,
                                                 (int)kernel_h,
                                                 padding: padding,
                                                 output_padding: out_pad_h,
                                                 stride: (int)stride_h,
                                                 dilation: (int)dilation_rate[0]);

            var out_width = conv_utils.deconv_output_length(width,
                                                (int)kernel_w,
                                                padding: padding,
                                                output_padding: out_pad_w,
                                                stride: (int)stride_w,
                                                dilation: (int)dilation_rate[1]);

            Tensor output_shape_tensor;
            if (data_format == "channels_first")
                output_shape_tensor = array_ops.stack(new object[] { batch_size, filters, out_height, out_width });
            else
                output_shape_tensor = array_ops.stack(new object[] { batch_size, out_height, out_width, filters });

            var outputs = keras.backend.conv2d_transpose(
                inputs,
                kernel,
                output_shape_tensor,
                strides: strides,
                padding: padding,
                data_format: data_format,
                dilation_rate: dilation_rate);

            if (!tf.Context.executing_eagerly())
            {
                var out_shape = ComputeOutputShape(inputs.shape);
                outputs.shape = out_shape;
            }

            if (use_bias)
                tf.nn.bias_add(
                    outputs,
                    bias,
                    data_format: conv_utils.convert_data_format(data_format, ndim: 4));

            if (activation != null)
                return activation.Apply(outputs);

            return outputs;
        }

        public override Shape ComputeOutputShape(Shape input_shape)
        {
            var output_shape = input_shape.dims;
            var (c_axis, h_axis, w_axis) = (3, 1, 2);
            if (data_format == "channels_first")
                (c_axis, h_axis, w_axis) = (1, 2, 3);

            var (kernel_h, kernel_w) = kernel_size;
            var (stride_h, stride_w) = strides;

            var (out_pad_h, out_pad_w) = (-1, -1);
            output_shape[c_axis] = filters;
            output_shape[h_axis] = conv_utils.deconv_output_length(
                (int)output_shape[h_axis],
                (int)kernel_h,
                padding: padding,
                output_padding: out_pad_h,
                stride: (int)stride_h,
                dilation: (int)dilation_rate[0]);
            output_shape[w_axis] = conv_utils.deconv_output_length(
                (int)output_shape[w_axis],
                (int)kernel_w,
                padding: padding,
                output_padding: out_pad_w,
                stride: (int)stride_w,
                dilation: (int)dilation_rate[1]);

            return new Shape(output_shape);
        }
    }
}
