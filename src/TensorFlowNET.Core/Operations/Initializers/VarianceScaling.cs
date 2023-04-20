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
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;

namespace Tensorflow.Operations.Initializers
{
    /// <summary>
    /// Initializer capable of adapting its scale to the shape of weights tensors.
    /// </summary>
    public class VarianceScaling : IInitializer
    {
        protected float _scale;
        protected string _mode;
        protected int? _seed;
        protected TF_DataType _dtype;
        protected string _distribution;
        private readonly Dictionary<string, object> _config;

        public virtual string ClassName => "VarianceScaling";

        public virtual IDictionary<string, object> Config => _config;

        public VarianceScaling(float scale = 1.0f,
            string mode = "fan_in",
            string distribution = "truncated_normal",
            int? seed = null,
            TF_DataType dtype = TF_DataType.TF_FLOAT)
        {
            if (!dtype.is_floating())
                throw new TypeError("Cannot create initializer for non-floating point type.");
            if (!new string[] { "fan_in", "fan_out", "fan_avg" }.Contains(mode))
                throw new TypeError($"Unknown {mode} %s [fan_in, fan_out, fan_avg]");
            if(distribution == "normal")
            {
                distribution = "truncated_normal";
            }
            if(!new string[] { "uniform", "truncated_normal", "untruncated_normal" }.Contains(distribution))
            {
                throw new ValueError($"Invalid `distribution` argument: {distribution}");
            }

            if (scale <= 0)
                throw new ValueError("`scale` must be positive float.");

            _scale = scale;
            _mode = mode;
            _seed = seed;
            _dtype = dtype;
            _distribution = distribution;

            _config = new();
            _config["scale"] = _scale;
            _config["mode"] = _mode;
            _config["distribution"] = _distribution;
            _config["seed"] = _seed;
        }

        public Tensor Apply(InitializerArgs args)
        {
            if (args.DType == TF_DataType.DtInvalid)
                args.DType = this._dtype;

            float n = 0;
            var (fan_in, fan_out) = _compute_fans(args.Shape);
            var scale = this._scale;
            if (_mode == "fan_in")
                scale /= Math.Max(1.0f, fan_in);
            else if (_mode == "fan_out")
                scale /= Math.Max(1.0f, fan_out);
            else
                scale /= Math.Max(1.0f, (fan_in + fan_out) / 2);

            if(_distribution == "truncated_normal")
            {
                var stddev = Math.Sqrt(scale) / .87962566103423978f;
                return random_ops.truncated_normal(args.Shape, 0.0f, (float)stddev, args.DType);
            }
            else if(_distribution == "untruncated_normal")
            {
                var stddev = Math.Sqrt(scale);
                return random_ops.random_normal(args.Shape, 0.0f, (float)stddev, args.DType);
            }
            else
            {
                var limit = (float)Math.Sqrt(scale * 3.0f);
                return random_ops.random_uniform(args.Shape, -limit, limit, args.DType);
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
