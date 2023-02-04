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
    public class Ones : IInitializer
    {
        private TF_DataType dtype;

        private readonly Dictionary<string, object> _config;

        public string ClassName => "Ones";
        public IDictionary<string, object> Config => new Dictionary<string, object>();

        public Ones(TF_DataType dtype = TF_DataType.TF_FLOAT)
        {
            this.dtype = dtype;
        }

        public Tensor Apply(InitializerArgs args)
        {
            if (args.DType == TF_DataType.DtInvalid)
                args.DType = this.dtype;

            return array_ops.ones(args.Shape, dtype);
        }
    }
}
