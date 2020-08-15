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
using System.Linq;

namespace Tensorflow.Operations
{
    public class _NonAtrousConvolution
    {
        public string padding;
        public string name;
        public int[] strides;
        public string data_format;
        private Func<Conv2dParams, Tensor> conv_op;

        public _NonAtrousConvolution(TensorShape input_shape,
            TensorShape filter_shape,
            string padding,
            string data_format,
            int[] strides,
            string name)
        {
            this.padding = padding;
            this.name = name;
            var conv_dims = input_shape.ndim - 2;
            if (conv_dims == 1)
            {
                throw new NotImplementedException("_NonAtrousConvolution conv_dims 1");
            }
            else if (conv_dims == 2)
            {
                var list = strides.ToList();

                if (string.IsNullOrEmpty(data_format) || data_format == "NHWC")
                {
                    data_format = "NHWC";
                    list.Insert(0, 1);
                    list.Add(1);
                }
                else if (data_format == "NCHW")
                    list.InsertRange(0, new int[] { 1, 1 });
                else
                    throw new ValueError("data_format must be \"NHWC\" or \"NCHW\".");

                strides = list.ToArray();
                this.strides = strides;
                this.data_format = data_format;
                conv_op = gen_nn_ops.conv2d;
            }
            else if (conv_dims == 3)
            {
                throw new NotImplementedException("_NonAtrousConvolution conv_dims 3");
            }
        }

        public Tensor __call__(Tensor inp, IVariableV1 filter)
        {
            return conv_op(new Conv2dParams
            {
                Input = inp,
                Filter = filter.AsTensor(),
                Strides = strides,
                Padding = padding,
                DataFormat = data_format,
                Name = name
            });
        }
    }
}
