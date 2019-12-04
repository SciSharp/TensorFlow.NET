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

namespace Tensorflow.Operations.Initializers
{
    public class GlorotUniform : VarianceScaling
    {
        public GlorotUniform(float scale = 1.0f,
            string mode = "FAN_AVG",
            bool uniform = true,
            int? seed = null,
            TF_DataType dtype = TF_DataType.TF_FLOAT) : base(factor: scale, 
                mode: mode, 
                uniform: uniform,
                seed: seed, 
                dtype: dtype)
        {

        }

        public object get_config()
        {
            return new
            {
                scale = _scale,
                mode = _mode,
                seed = _seed,
                dtype = _dtype
            };
        }
    }
}
