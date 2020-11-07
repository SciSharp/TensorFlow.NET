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
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class RefVariable
    {
        public static Tensor operator +(RefVariable x, int y) => op_helper("add", x, y);
        public static Tensor operator +(RefVariable x, float y) => op_helper("add", x, y);
        public static Tensor operator +(RefVariable x, double y) => op_helper("add", x, y);

        public static Tensor operator -(RefVariable x, int y) => op_helper("sub", x, y);
        public static Tensor operator -(RefVariable x, float y) => op_helper("sub", x, y);
        public static Tensor operator -(RefVariable x, double y) => op_helper("sub", x, y);
        public static Tensor operator -(RefVariable x, Tensor y) => op_helper("sub", x, y);

        public static Tensor operator <(RefVariable x, Tensor y) => gen_math_ops.less(x.value(), y);

        public static Tensor operator >(RefVariable x, Tensor y) => gen_math_ops.greater(x.value(), y);

        private static Tensor op_helper<T>(string default_name, RefVariable x, T y)
        {
            var xVal = x.value();
            return tf_with(ops.name_scope(null, default_name, new { xVal, y }), scope =>
            {
                string name = scope;
                var yTensor = ops.convert_to_tensor(y, xVal.dtype.as_base_dtype(), "y");
                Tensor result = null;
                switch (default_name)
                {
                    case "add":
                        result = gen_math_ops.add(xVal, yTensor, name);
                        break;
                    case "sub":
                        result = gen_math_ops.sub(xVal, yTensor, name);
                        break;
                    default:
                        throw new NotImplementedException("");
                }
                return result;
            });
        }
    }
}
