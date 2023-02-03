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
    public class Constant<T> : IInitializer
    {
        TF_DataType dtype;
        T value;
        bool _verify_shape;

        private readonly Dictionary<string, object> _config;

        public string ClassName => "Constant";
        public IDictionary<string, object> Config => _config;

        public Constant(T value, TF_DataType dtype = TF_DataType.TF_FLOAT, bool verify_shape = false)
        {
            this.value = value;
            this.dtype = dtype;
            _verify_shape = verify_shape;

            _config = new Dictionary<string, object>();
            _config["value"] = this.value;
        }

        public Tensor Apply(InitializerArgs args)
        {
            if (args.DType == TF_DataType.DtInvalid)
                args.DType = this.dtype;

            args.VerifyShape = _verify_shape;

            return constant_op.constant(value, args.DType, args.Shape,
                name: "Const",
                verify_shape: args.VerifyShape,
                allow_broadcast: false);
        }
    }
}
