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

using System;
using System.Linq;

namespace Tensorflow.Operations
{
    public class _WithSpaceToBatch
    {
        private _NonAtrousConvolution call;

        public _WithSpaceToBatch(TensorShape input_shape,
            int[] dilation_rate,
            string padding,
            Func<int, string, _NonAtrousConvolution> build_op,
            TensorShape filter_shape = null,
            int[] spatial_dims = null,
            string data_format = null)
        {
            var dilation_rate_tensor = ops.convert_to_tensor(dilation_rate, TF_DataType.TF_INT32, name: "dilation_rate");
            var rate_shape = dilation_rate_tensor.TensorShape;
            var num_spatial_dims = rate_shape.dims[0];
#pragma warning disable CS0219 // Variable is assigned but its value is never used
            int starting_spatial_dim = -1;
#pragma warning restore CS0219 // Variable is assigned but its value is never used
            if (!string.IsNullOrEmpty(data_format) && data_format.StartsWith("NC"))
                starting_spatial_dim = 2;
            else
                starting_spatial_dim = 1;

            if (spatial_dims == null)
                throw new NotImplementedException("_WithSpaceToBatch spatial_dims");

            var orig_spatial_dims = spatial_dims;
            spatial_dims = spatial_dims.OrderBy(x => x).ToArray();
            if (!Enumerable.SequenceEqual(spatial_dims, orig_spatial_dims) || spatial_dims.Any(x => x < 1))
                throw new ValueError("spatial_dims must be a montonically increasing sequence of positive integers");

            int expected_input_rank = -1;
            if (!string.IsNullOrEmpty(data_format) && data_format.StartsWith("NC"))
                expected_input_rank = spatial_dims.Last();
            else
                expected_input_rank = spatial_dims.Last() + 1;

            var const_rate = tensor_util.constant_value(dilation_rate_tensor);
            var rate_or_const_rate = dilation_rate;
            if(!(const_rate is null))
            {
                if (const_rate.Data<int>().Count(x => x == 1) == const_rate.size)
                {
                    call = build_op(num_spatial_dims, padding);
                    return;
                }
            }
        }

        public Tensor __call__(Tensor inp, RefVariable filter)
        {
            return call.__call__(inp, filter);
        }
    }
}
