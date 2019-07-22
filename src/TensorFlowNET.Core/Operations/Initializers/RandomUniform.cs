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

namespace Tensorflow.Operations.Initializers
{
    public class RandomUniform : IInitializer
    {
        private int? seed;
        private float minval;
        private float maxval;
        private TF_DataType dtype;

        public RandomUniform()
        {

        }

        public Tensor call(TensorShape shape, TF_DataType dtype = TF_DataType.DtInvalid)
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
