﻿/*****************************************************************************
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

namespace Tensorflow.Operations
{
    public class Conv2dParams
    {
        public string Name { get; set; }

        /// <summary>
        /// An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
        /// Specify the data format of the input and output data. With the
        /// default format "NHWC", the data is stored in the order of:
        /// [batch, height, width, channels].
        /// </summary>
        public string DataFormat { get; set; } = "NHWC";

        /// <summary>
        /// Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
        /// A 4-D tensor. The dimension order is interpreted according to the value
        /// </summary>
        public Tensor Input { get; set; }

        /// <summary>
        /// An integer vector representing the shape of `input`
        /// </summary>
        public Tensor InputSizes { get; set; }

        /// <summary>
        /// A 4-D tensor of shape
        /// </summary>
        public IVariableV1 Filter { get; set; }

        /// <summary>
        /// An integer vector representing the tensor shape of `filter`
        /// </summary>
        public Tensor FilterSizes { get; set; }

        /// <summary>
        /// A `Tensor`. Must have the same type as `filter`.
        /// 4-D with shape `[batch, out_height, out_width, out_channels]`.
        /// </summary>
        public Tensor OutBackProp { get; set; }

        /// <summary>
        /// The stride of the sliding window for each
        /// dimension of `input`. The dimension order is determined by the value of
        /// `data_format`, see below for details.
        /// </summary>
        public int[] Strides { get; set; }

        /// <summary>
        /// A `string` from: `"SAME", "VALID", "EXPLICIT"`.
        /// </summary>
        public string Padding { get; set; }

        public int[] ExplicitPaddings { get; set; } = new int[0];

        public bool UseCudnnOnGpu { get; set; } = true;

        public int[] Dilations { get; set; } = new int[] { 1, 1, 1, 1 };

        public Conv2dParams()
        {

        }
    }
}
