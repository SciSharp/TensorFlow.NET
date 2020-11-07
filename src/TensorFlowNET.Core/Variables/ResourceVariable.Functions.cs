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
    public partial class ResourceVariable
    {
        /// <summary>
        /// Subtracts a value from this variable.
        /// </summary>
        /// <param name="delta"></param>
        /// <param name="use_locking"></param>
        /// <param name="name"></param>
        /// <param name="read_value"></param>
        public void assign_sub(Tensor delta, bool use_locking = false, string name = null, bool read_value = true)
        {
            gen_resource_variable_ops.assign_sub_variable_op(handle, delta, name: name);
        }

        /// <summary>
        /// Adds a value to this variable.
        /// </summary>
        /// <param name="delta"></param>
        /// <param name="use_locking"></param>
        /// <param name="name"></param>
        /// <param name="read_value"></param>
        public void assign_add(Tensor delta, bool use_locking = false, string name = null, bool read_value = true)
        {
            gen_resource_variable_ops.assign_add_variable_op(handle, delta, name: name);
        }
    }
}
