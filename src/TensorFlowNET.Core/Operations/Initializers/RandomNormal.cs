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
using System.Text;

namespace Tensorflow.Operations.Initializers
{
    public class RandomNormal : IInitializer
    {
        private float mean;
        private float stddev;
        private int? seed;
        private TF_DataType dtype;

        public RandomNormal(float mean = 0.0f,
            float stddev = 1.0f,
            int? seed = null,
            TF_DataType dtype = TF_DataType.TF_FLOAT)
        {
            this.mean = mean;
            this.stddev = stddev;
            this.seed = seed;
            this.dtype = dtype;
        }

        public Tensor call(TensorShape shape, TF_DataType dtype = TF_DataType.DtInvalid)
        {
            throw new NotImplementedException();
        }

        public object get_config()
        {
            throw new NotImplementedException();
        }
    }
}
