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
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.ArgsDefinition;
using static Tensorflow.Binding;

namespace Tensorflow.Operations
{
    public class ConvolutionInternal
    {
        ConvolutionalArgs args;

        string data_format => args.DataFormat;
        string name;
        string padding => args.Padding;

        public ConvolutionInternal(ConvolutionalArgs args)
        {
            this.args = args;
            name = args.Name;
        }

        public Tensor Apply(Tensors input, IVariableV1 filters)
        {
            var filters_rank = filters.shape.rank;
            var inputs_rank = input.shape.rank;
            var num_spatial_dims = args.NumSpatialDims;
            if (num_spatial_dims == Unknown)
                num_spatial_dims = filters_rank - 2;

            // Channel dimension.
            var num_batch_dims = inputs_rank - num_spatial_dims - 1;
            if (!new[] { 1, 2, 3 }.Contains(num_spatial_dims))
                throw new ValueError($"num_spatial_dims (input.shape.ndims - num_batch_dims - 1) must be one " +
                    $"of 1, 2 or 3 but saw {num_spatial_dims}. num_batch_dims: {num_batch_dims}.");

            var channel_index = num_batch_dims + num_spatial_dims;
            var dilations = _get_sequence(args.DilationRate, num_spatial_dims, channel_index);
            var strides = _get_sequence(args.Strides, num_spatial_dims, channel_index);

            Tensor result = null;
            tf_with(ops.name_scope(name, default_name: null), scope =>
            {
                name = scope;
                if (num_spatial_dims == 2)
                {
                    result = gen_nn_ops.conv2d(new Conv2dParams
                    {
                        Input = input,
                        Filter = filters,
                        Strides = strides,
                        Padding = padding,
                        DataFormat = data_format,
                        Dilations = dilations,
                        Name = name
                    });
                }
                else
                    throw new NotImplementedException("");
            });

            return result;
        }

        int[] _get_sequence(int[] value, int n, int channel_index)
        {
            var seq = new List<int>();

            if (channel_index == 1)
            {
                seq.Add(1);
                seq.Add(1);
                seq.AddRange(value);
            }
            else
            {
                seq.Add(1);
                seq.AddRange(value);
                seq.Add(1);
            }

            return seq.ToArray();
        }
    }
}
