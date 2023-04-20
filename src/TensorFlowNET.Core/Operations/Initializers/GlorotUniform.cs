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

using System.Collections.Generic;

namespace Tensorflow.Operations.Initializers
{
    public class GlorotUniform : VarianceScaling
    {
        private readonly Dictionary<string, object> _config;

        public override string ClassName => "GlorotUniform";
        public override IDictionary<string, object> Config => _config;

        public GlorotUniform(float scale = 1.0f,
            string mode = "fan_avg",
            string distribution = "uniform",
            int? seed = null,
            TF_DataType dtype = TF_DataType.TF_FLOAT) : base(scale: scale,
                mode: mode,
                distribution: distribution,
                seed: seed,
                dtype: dtype)
        {
            _config = new Dictionary<string, object>();
            _config["seed"] = _seed;
        }
    }
}
