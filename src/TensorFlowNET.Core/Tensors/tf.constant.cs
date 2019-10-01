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

namespace Tensorflow
{
    public partial class tensorflow
    {
        // public static Tensor constant(NDArray nd, string name = "Const") => constant_op.constant(nd, name: name);

        public Tensor constant(object value,
            TF_DataType dtype = TF_DataType.DtInvalid,
            int[] shape = null,
            string name = "Const",
            bool verify_shape = false) => constant_op._constant_impl(value,
                dtype,
                shape,
                name,
                verify_shape: verify_shape,
                allow_broadcast: false);

        public Tensor constant(string value,
            string name = "Const") => constant_op._constant_impl(value,
                @string,
                new int[] { 1 },
                name,
                verify_shape: false,
                allow_broadcast: false);

        public Tensor constant(float value,
            int shape,
            string name = "Const") => constant_op._constant_impl(value,
                float32,
                new int[] { shape },
                name,
                verify_shape: false,
                allow_broadcast: false);

        public Tensor zeros(TensorShape shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null) 
            => array_ops.zeros(shape, dtype, name);

        public Tensor ones(TensorShape shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null) 
            => array_ops.ones(shape, dtype, name);

        public Tensor size(Tensor input,
            string name = null,
            TF_DataType out_type = TF_DataType.TF_INT32) => array_ops.size(input,
                name,
                optimize: true,
                out_type: out_type);
    }
}
