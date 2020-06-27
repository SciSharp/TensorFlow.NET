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

using NumSharp;
using System;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class ResourceVariable
    {
        public static Tensor operator +(ResourceVariable x, int y) => op_helper("add", x, y);
        public static Tensor operator +(ResourceVariable x, float y) => op_helper("add", x, y);
        public static Tensor operator +(ResourceVariable x, double y) => op_helper("add", x, y);
        public static Tensor operator +(ResourceVariable x, ResourceVariable y) => op_helper("add", x, y);
        public static Tensor operator -(ResourceVariable x, int y) => op_helper("sub", x, y);
        public static Tensor operator -(ResourceVariable x, float y) => op_helper("sub", x, y);
        public static Tensor operator -(ResourceVariable x, double y) => op_helper("sub", x, y);
        public static Tensor operator -(ResourceVariable x, Tensor y) => op_helper("sub", x, y);
        public static Tensor operator -(ResourceVariable x, ResourceVariable y) => op_helper("sub", x, y);

        public static Tensor operator *(ResourceVariable x, ResourceVariable y) => op_helper("mul", x, y);
        public static Tensor operator *(ResourceVariable x, NDArray y) => op_helper("mul", x, y);

        public static Tensor operator <(ResourceVariable x, Tensor y) => op_helper("less", x, y);

        public static Tensor operator >(ResourceVariable x, Tensor y) => op_helper("greater", x, y);

        private static Tensor op_helper<T>(string default_name, ResourceVariable x, T y)
            => tf_with(ops.name_scope(null, default_name, new { x, y }), scope =>
            {
                string name = scope;
                var xVal = x.value();
                var yTensor = ops.convert_to_tensor(y, xVal.dtype.as_base_dtype(), "y");
                Tensor result = null;
                switch (default_name)
                {
                    case "add":
                        result = x.dtype == TF_DataType.TF_STRING ?
                            gen_math_ops.add(xVal, yTensor, name) :
                            gen_math_ops.add_v2(xVal, yTensor, name);
                        break;
                    case "sub":
                        result = gen_math_ops.sub(xVal, yTensor, name);
                        break;
                    case "mul":
                        result = gen_math_ops.mul(xVal, yTensor, name: name);
                        break;
                    case "less":
                        result = gen_math_ops.less(xVal, yTensor, name);
                        break;
                    case "greater":
                        result = gen_math_ops.greater(xVal, yTensor, name);
                        break;
                    default:
                        throw new NotImplementedException("");
                }

                // x.assign(result);
                // result.ResourceVar = x;
                return result;
            });
    }
}
