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
    public class Zeros : IInitializer
    {
        Shape shape;
        TF_DataType dtype;

        public string ClassName => "Zeros";
        public IDictionary<string, object> Config => new Dictionary<string, object>();

        public Zeros(Shape shape = null, TF_DataType dtype = TF_DataType.TF_FLOAT)
        {
            this.shape = shape;
            this.dtype = dtype;
        }

        public Tensor Apply(InitializerArgs args)
        {
            if (args.DType == TF_DataType.DtInvalid)
                args.DType = dtype;
            if (args.Shape == null)
                args.Shape = shape;

            return array_ops.zeros(args.Shape, dtype);
        }
    }
}
