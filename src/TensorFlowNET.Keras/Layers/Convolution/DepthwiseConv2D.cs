using System;
using System.Collections.Generic;
using System.Text;
using System;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Saving;
using Tensorflow.Common.Types;
using Tensorflow.Keras.Utils;
using Tensorflow.Operations;
using Newtonsoft.Json;
using System.Security.Cryptography;

namespace Tensorflow.Keras.Layers
{
    public class DepthwiseConv2DArgs: Conv2DArgs
    {
        /// <summary>
        /// depth_multiplier: The number of depthwise convolution output channels for
        /// each input channel.The total number of depthwise convolution output
        /// channels will be equal to `filters_in* depth_multiplier`.
        /// </summary>
        [JsonProperty("depth_multiplier")]
        public int DepthMultiplier { get; set; } = 1;

        [JsonProperty("depthwise_initializer")]
        public IInitializer DepthwiseInitializer { get; set; }
    }

    public class DepthwiseConv2D : Conv2D
    {
        /// <summary>
        /// depth_multiplier: The number of depthwise convolution output channels for
        /// each input channel.The total number of depthwise convolution output
        /// channels will be equal to `filters_in* depth_multiplier`.
        /// </summary>
        int DepthMultiplier = 1;
        
        IInitializer DepthwiseInitializer;

        int[] strides;

        int[] dilation_rate;

        string getDataFormat()
        {
            return data_format == "channels_first" ? "NCHW" : "NHWC";
        }

        static int _id = 1;

        public DepthwiseConv2D(DepthwiseConv2DArgs args):base(args)
        {
            args.Padding = args.Padding.ToUpper();

            if(string.IsNullOrEmpty(args.Name))
                name = "DepthwiseConv2D_" + _id;

            this.DepthMultiplier = args.DepthMultiplier;
            this.DepthwiseInitializer = args.DepthwiseInitializer;

        }

        public override void build(KerasShapesWrapper input_shape)
        {
            //base.build(input_shape);

            var shape = input_shape.ToSingleShape();

            int channel_axis = data_format == "channels_first" ? 1 : -1;
            var input_channel = channel_axis < 0 ?
                shape.dims[shape.ndim + channel_axis] :
                shape.dims[channel_axis];

            var arg = args as DepthwiseConv2DArgs;

            if (arg.Strides.ndim != shape.ndim)
            {
                if (arg.Strides.ndim == 2)
                {
                    this.strides = new int[] { 1, (int)arg.Strides[0], (int)arg.Strides[1], 1 };
                }
                else
                {
                    this.strides = conv_utils.normalize_tuple(new int[] { (int)arg.Strides[0] }, shape.ndim, "strides");
                }
            }
            else
            {
                this.strides = arg.Strides.dims.Select(o=>(int)(o)).ToArray();
            }

            if (arg.DilationRate.ndim != shape.ndim)
            {
                this.dilation_rate = conv_utils.normalize_tuple(new int[] { (int)arg.DilationRate[0] }, shape.ndim, "dilation_rate");
            }

            long channel_data = data_format == "channels_first" ? shape[0] : shape[shape.Length - 1];

            var depthwise_kernel_shape = this.kernel_size.dims.concat(new long[] {
                channel_data,
                this.DepthMultiplier
            });

            this.kernel = this.add_weight(
                shape: depthwise_kernel_shape,
                initializer: this.DepthwiseInitializer != null ? this.DepthwiseInitializer : this.kernel_initializer,
                name: "depthwise_kernel",
                trainable: true,
                dtype: DType,
                regularizer: this.kernel_regularizer
            );

            var axes = new Dictionary<int, int>();
            axes.Add(-1, (int)input_channel);
            inputSpec = new InputSpec(min_ndim: rank + 2, axes: axes);


            if (use_bias)
            {
                bias = add_weight(name: "bias",
                    shape: ((int)channel_data),
                    initializer: bias_initializer,
                    trainable: true,
                    dtype: DType);
            }

            built = true;
            _buildInputShape = input_shape;
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null,
            bool? training = false, IOptionalArgs? optional_args = null)
        {
            Tensor outputs = null;

            outputs = gen_nn_ops.depthwise_conv2d_native(
                    inputs,
                    filter: this.kernel.AsTensor(),
                    strides: this.strides,
                    padding: this.padding,
                    dilations: this.dilation_rate,
                    data_format: this.getDataFormat(),
                    name: name
                );

            if (use_bias)
            {
                if (data_format == "channels_first")
                {
                    throw new NotImplementedException("call channels_first");
                }
                else
                {
                    outputs = gen_nn_ops.bias_add(outputs, ops.convert_to_tensor(bias),
                        data_format: this.getDataFormat(), name: name);
                }
            }

            if (activation != null)
                outputs = activation.Apply(outputs);


            return outputs;
        }

    }
}