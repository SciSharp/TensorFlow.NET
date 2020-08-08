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

using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Optimizers;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public KerasOptimizers optimizers => new KerasOptimizers();

        public class KerasOptimizers
        {
            public SGD SGD(float learning_rate) => new SGD(learning_rate);

            public Adam Adam(float learning_rate = 0.001f,
                float beta_1 = 0.9f,
                float beta_2 = 0.999f,
                float epsilon = 1e-7f,
                bool amsgrad = false,
                string name = "Adam") => new Adam(learning_rate: learning_rate,
                    beta_1: beta_1,
                    beta_2: beta_2,
                    epsilon: epsilon,
                    amsgrad: amsgrad,
                    name: name);
        }
    }
}
