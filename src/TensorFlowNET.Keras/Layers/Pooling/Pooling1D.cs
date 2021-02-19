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

using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;

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

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool? training = null)
        {
            int[] pool_shape;
            int[] strides;
            if (args.DataFormat == "channels_last")
            {
                pool_shape = new int[] { 1, args.PoolSize, 1 };
                strides = new int[] { 1, args.Strides, 1 };
            }
            else
            {
                pool_shape = new int[] { 1, 1, args.PoolSize };
                strides = new int[] { 1, 1, args.Strides };
            }

            var outputs = args.PoolFunction.Apply(
                inputs,
                ksize: pool_shape,
                strides: strides,
                padding: args.Padding.ToUpper(),
                data_format: conv_utils.convert_data_format(args.DataFormat, 3));

            return outputs;
        }
    }
}
