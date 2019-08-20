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

using Tensorflow.Operations.Initializers;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public IInitializer zeros_initializer => new Zeros();
        public IInitializer ones_initializer => new Ones();
        public IInitializer glorot_uniform_initializer => new GlorotUniform();
        public IInitializer uniform_initializer => new RandomUniform();

        public variable_scope variable_scope(string name,
               string default_name = null,
               Tensor[] values = null,
               bool? reuse = null,
               bool auxiliary_name_scope = true) => new variable_scope(name,
                   default_name,
                   values,
                   reuse: reuse,
                   auxiliary_name_scope: auxiliary_name_scope);

        public variable_scope variable_scope(VariableScope scope,
              string default_name = null,
              Tensor[] values = null,
              bool? reuse = null,
              bool auxiliary_name_scope = true) => new variable_scope(scope,
                  default_name,
                  values,
                  reuse: reuse,
                  auxiliary_name_scope: auxiliary_name_scope);

        public IInitializer truncated_normal_initializer(float mean = 0.0f,
            float stddev = 1.0f,
            int? seed = null,
            TF_DataType dtype = TF_DataType.DtInvalid) => new TruncatedNormal(mean: mean,
                stddev: stddev,
                seed: seed,
                dtype: dtype);
    }
}
