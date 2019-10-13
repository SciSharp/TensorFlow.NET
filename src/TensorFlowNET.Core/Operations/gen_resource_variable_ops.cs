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
    public static class gen_resource_variable_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        public static Operation assign_variable_op(Tensor resource, Tensor value, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("AssignVariableOp", name, new { resource, value });

            return _op;
        }

        public static Tensor var_is_initialized_op(Tensor resource, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("VarIsInitializedOp", name, new { resource });

            return _op;
        }

        /// <summary>
        /// Creates a handle to a Variable resource.
        /// </summary>
        /// <param name="dtype"></param>
        /// <param name="shape"></param>
        /// <param name="container"></param>
        /// <param name="shared_name"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor var_handle_op(TF_DataType dtype, TensorShape shape, 
            string container ="", string shared_name = "", string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("VarHandleOp", name, new {
                dtype,
                shape,
                container,
                shared_name
            });

            return _op;
        }
    }
}
