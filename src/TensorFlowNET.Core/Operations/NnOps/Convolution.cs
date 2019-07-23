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

namespace Tensorflow.Operations
{
    public class Convolution
    {
        public TensorShape input_shape;
        public TensorShape filter_shape;
        public string data_format;
        public int[] strides;
        public string name;
        public _WithSpaceToBatch conv_op;

        public Convolution(TensorShape input_shape,
            TensorShape filter_shape,
            string padding,
            int[] strides,
            int[] dilation_rate,
            string name = null,
            string data_format = null)
        {
            var num_total_dims = filter_shape.NDim;
            var num_spatial_dims = num_total_dims - 2;
            int input_channels_dim;
            int[] spatial_dims;
            if (string.IsNullOrEmpty(data_format) || !data_format.StartsWith("NC"))
            {
                input_channels_dim = input_shape.Dimensions[num_spatial_dims + 1];
                spatial_dims = Enumerable.Range(1, num_spatial_dims).ToArray();
            }
            else
            {
                input_channels_dim = input_shape.Dimensions[1];
                spatial_dims = Enumerable.Range(2, num_spatial_dims).ToArray();
            }

            this.input_shape = input_shape;
            this.filter_shape = filter_shape;
            this.data_format = data_format;
            this.strides = strides;
            this.name = name;

            conv_op = new _WithSpaceToBatch(
                input_shape,
                dilation_rate: dilation_rate,
                padding: padding,
                build_op: _build_op,
                filter_shape: filter_shape,
                spatial_dims: spatial_dims,
                data_format: data_format);
        }

        public _NonAtrousConvolution _build_op(int _, string padding)
        {
            return new _NonAtrousConvolution(input_shape,
                filter_shape: filter_shape,
                padding: padding,
                data_format: data_format,
                strides: strides,
                name: name);
        }

        public Tensor __call__(Tensor inp, RefVariable filter)
        {
            return conv_op.__call__(inp, filter);
        }
    }
}
