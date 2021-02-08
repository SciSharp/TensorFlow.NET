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
    public class Pooling2D : Layer
    {
        Pooling2DArgs args;
        InputSpec input_spec;

        public Pooling2D(Pooling2DArgs args)
            : base(args)
        {
            this.args = args;
            args.PoolSize = conv_utils.normalize_tuple(args.PoolSize, 2, "pool_size");
            args.Strides = conv_utils.normalize_tuple(args.Strides ?? args.PoolSize, 2, "strides");
            args.Padding = conv_utils.normalize_padding(args.Padding);
            args.DataFormat = conv_utils.normalize_data_format(args.DataFormat);
            input_spec = new InputSpec(ndim: 4);
        }

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool? training = null)
        {
            int[] pool_shape;
            int[] strides;
            if (args.DataFormat == "channels_last")
            {
                pool_shape = new int[] { 1, args.PoolSize.dims[0], args.PoolSize.dims[1], 1 };
                strides = new int[] { 1, args.Strides.dims[0], args.Strides.dims[1], 1 };
            }
            else
            {
                pool_shape = new int[] { 1, 1, args.PoolSize.dims[0], args.PoolSize.dims[1] };
                strides = new int[] { 1, 1, args.Strides.dims[0], args.Strides.dims[1] };
            }

            var outputs = args.PoolFunction.Apply(
                inputs,
                ksize: pool_shape,
                strides: strides,
                padding: args.Padding.ToUpper(),
                data_format: conv_utils.convert_data_format(args.DataFormat, 4));

            return outputs;
        }
    }
}
