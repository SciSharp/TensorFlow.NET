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

namespace Tensorflow
{
    public partial class tensorflow
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="value"></param>
        /// <param name="dtype"></param>
        /// <param name="shape"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor constant(object value,
            TF_DataType dtype = TF_DataType.DtInvalid,
            TensorShape shape = null,
            string name = "Const")
            => constant_op._constant_impl(value,
                dtype,
                shape,
                name,
                verify_shape: false,
                allow_broadcast: true);

        public Tensor zeros(TensorShape shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null)
            => array_ops.zeros(shape, dtype, name);

        public Tensor zeros(Tensor shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null)
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
