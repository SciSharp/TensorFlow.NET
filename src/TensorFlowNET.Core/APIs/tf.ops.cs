﻿/*****************************************************************************
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
    public static partial class tf
    {
        public static Tensor assign(Tensor @ref, object value, bool validate_shape = true, bool use_locking = true, string name = null) 
            => state_ops.assign(@ref, value, validate_shape, use_locking, name);

        public static object get_collection(string key, string scope = "") 
            => get_default_graph().get_collection(key, scope: scope);

        /// <summary>
        /// Returns a context manager that creates hierarchical names for operations.
        /// </summary>
        /// <param name="name">The name argument that is passed to the op function.</param>
        /// <param name="default_name">The default name to use if the name argument is None.</param>
        /// <param name="values">The list of Tensor arguments that are passed to the op function.</param>
        /// <returns>The scope name.</returns>
        public static ops.NameScope name_scope(string name, string default_name = "", object values = null) 
            => new ops.NameScope(name, default_name, values);
    }
}
