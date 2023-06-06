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
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;
using Tensorflow.Common.Types;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    public class Pooling1D : Layer
    {
        Pooling1DArgs args;
        InputSpec input_spec;

        public Pooling1D(Pooling1DArgs args)
            : base(args)
        {
            this.args = args;
            args.Padding = conv_utils.normalize_padding(args.Padding);
            args.DataFormat = conv_utils.normalize_data_format(args.DataFormat);
            input_spec = new InputSpec(ndim: 3);
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            int pad_axis = args.DataFormat == "channels_first" ? 2 : 3;
            inputs = tf.expand_dims(inputs, pad_axis);
            int[] pool_shape = new int[] { args.PoolSize, 1 };
            int[] strides = new int[] { args.Strides, 1 };
            var ndim = inputs[0].ndim;

            if (args.DataFormat == "channels_last")
            {
                pool_shape = new int[] { 1 }.Concat(pool_shape).Concat(new int[] { 1 }).ToArray();
                strides = new int[] { 1 }.Concat(strides).Concat(new int[] { 1 }).ToArray();
            }
            else
            {
                pool_shape = new int[] { 1, 1 }.Concat(pool_shape).ToArray();
                strides = new int[] { 1, 1 }.Concat(strides).ToArray();
            }

            var outputs = args.PoolFunction.Apply(
                inputs,
                ksize: pool_shape,
                strides: strides,
                padding: args.Padding.ToUpper(),
                data_format: conv_utils.convert_data_format(args.DataFormat, ndim));

            return tf.squeeze(outputs, pad_axis);
        }
    }
}
