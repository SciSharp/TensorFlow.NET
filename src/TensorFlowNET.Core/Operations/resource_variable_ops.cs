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
    /// <summary>
    /// tensorflow\python\ops\resource_variable_ops.py
    /// </summary>
    public class resource_variable_ops
    {
        public static ITensorOrOperation shape_safe_assign_variable_handle(Tensor handle, int[] shape, Tensor value, string name = null)
        {
            var value_tensor = ops.convert_to_tensor(value);
            return gen_resource_variable_ops.assign_variable_op(handle,
                                                      value_tensor,
                                                      name: name);
        }

        public static bool is_resource_variable(VariableV1 var)
        {
            return var is ResourceVariable;
        }
    }
}
