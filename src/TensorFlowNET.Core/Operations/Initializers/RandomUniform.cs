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
    public class RandomUniform : IInitializer
    {
#pragma warning disable CS0649 // Field 'RandomUniform.seed' is never assigned to, and will always have its default value
        private int? seed;
#pragma warning restore CS0649 // Field 'RandomUniform.seed' is never assigned to, and will always have its default value
#pragma warning disable CS0649 // Field 'RandomUniform.minval' is never assigned to, and will always have its default value 0
        private float minval;
#pragma warning restore CS0649 // Field 'RandomUniform.minval' is never assigned to, and will always have its default value 0
#pragma warning disable CS0649 // Field 'RandomUniform.maxval' is never assigned to, and will always have its default value 0
        private float maxval;
#pragma warning restore CS0649 // Field 'RandomUniform.maxval' is never assigned to, and will always have its default value 0
#pragma warning disable CS0649 // Field 'RandomUniform.dtype' is never assigned to, and will always have its default value
        private TF_DataType dtype;
#pragma warning restore CS0649 // Field 'RandomUniform.dtype' is never assigned to, and will always have its default value

        public RandomUniform()
        {

        }

        public Tensor call(TensorShape shape, TF_DataType dtype = TF_DataType.DtInvalid, bool? verify_shape = null)
        {
            return random_ops.random_uniform(shape, 
                minval: minval, 
                maxval: maxval, 
                dtype: dtype, 
                seed: seed);
        }

        public object get_config()
        {
            return new {
                minval,
                maxval,
                seed,
                dtype
            };
        }
    }
}
