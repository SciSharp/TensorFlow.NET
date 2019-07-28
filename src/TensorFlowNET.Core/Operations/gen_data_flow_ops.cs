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
    public class gen_data_flow_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        public static Tensor dynamic_stitch(Tensor[] indices, Tensor[] data, string name = null)
        {
            var _attr_N = indices.Length;
            var _op = _op_def_lib._apply_op_helper("DynamicStitch", name, new { indices, data });

            return _op.outputs[0];
        }

        public static (Tensor, Tensor) tensor_array_v3(Tensor size, TF_DataType dtype = TF_DataType.DtInvalid, 
            int[] element_shape = null, bool dynamic_size = false, bool clear_after_read = true, 
            bool identical_element_shapes = false, string tensor_array_name = "tensor_array_name", string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("TensorArrayV3", name, new
            {
                size,
                dtype,
                element_shape,
                dynamic_size,
                clear_after_read,
                identical_element_shapes,
                tensor_array_name
            });

            return (null, null);
        }
    }
}
