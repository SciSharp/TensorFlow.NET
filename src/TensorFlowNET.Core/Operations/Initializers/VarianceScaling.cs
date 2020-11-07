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

namespace Tensorflow.Operations.Initializers
{
    /// <summary>
    /// Initializer capable of adapting its scale to the shape of weights tensors.
    /// </summary>
    public class VarianceScaling : IInitializer
    {
        protected float _scale;
        protected string _mode;
        protected string _distribution;
        protected int? _seed;
        protected TF_DataType _dtype;
        protected bool _uniform;

        public VarianceScaling(float factor = 2.0f,
            string mode = "FAN_IN",
            bool uniform = false,
            int? seed = null,
            TF_DataType dtype = TF_DataType.TF_FLOAT)
        {
            if (!dtype.is_floating())
                throw new TypeError("Cannot create initializer for non-floating point type.");
            if (!new string[] { "FAN_IN", "FAN_OUT", "FAN_AVG" }.Contains(mode))
                throw new TypeError($"Unknown {mode} %s [FAN_IN, FAN_OUT, FAN_AVG]");

            if (factor < 0)
                throw new ValueError("`scale` must be positive float.");

            _scale = factor;
            _mode = mode;
            _seed = seed;
            _dtype = dtype;
            _uniform = uniform;
        }

        public Tensor Apply(InitializerArgs args)
        {
            if (args.DType == TF_DataType.DtInvalid)
                args.DType = this._dtype;

            float n = 0;
            var (fan_in, fan_out) = _compute_fans(args.Shape);
            if (_mode == "FAN_IN")
                n = fan_in;
            else if (_mode == "FAN_OUT")
                n = fan_out;
            else if (_mode == "FAN_AVG")
                n = (fan_in + fan_out) / 2.0f;

            if (_uniform)
            {
                var limit = Convert.ToSingle(Math.Sqrt(3.0f * _scale / n));
                return random_ops.random_uniform(args.Shape, -limit, limit, args.DType);
            }
            else
            {
                var trunc_stddev = Convert.ToSingle(Math.Sqrt(1.3f * _scale / n));
                return random_ops.truncated_normal(args.Shape, 0.0f, trunc_stddev, args.DType,
                                                   seed: _seed);
            }
        }

        private (int, int) _compute_fans(int[] shape)
        {
            if (shape.Length < 1)
                return (1, 1);
            if (shape.Length == 1)
                return (shape[0], shape[0]);
            if (shape.Length == 2)
                return (shape[0], shape[1]);
            else
            {
                // Assuming convolution kernels (2D, 3D, or more).
                // kernel shape: (..., input_depth, depth)
                int receptive_field_size = 1;
                foreach (var dim in shape.Take(shape.Length - 2))
                    receptive_field_size *= dim;
                var fan_in = shape[shape.Length - 2] * receptive_field_size;
                var fan_out = shape[shape.Length - 1] * receptive_field_size;
                return (fan_in, fan_out);
            }
        }
    }
}
