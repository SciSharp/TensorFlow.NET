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
using System.Collections.Generic;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public void add_to_collection<T>(string name, T value)
            => get_default_graph().add_to_collection(name, value);

        public void add_to_collections<T>(List<string> names, T value)
            => get_default_graph().add_to_collections(names, value);

        public (Tensors, Tensor) clip_by_global_norm(Tensor[] t_list, float clip_norm, Tensor use_norm = null, string name = null)
            => clip_ops.clip_by_global_norm(t_list, clip_norm, use_norm: use_norm, name: name);

        public Tensor assign(IVariableV1 @ref, object value, bool validate_shape = true, bool use_locking = true, string name = null)
            => state_ops.assign(@ref, value, validate_shape, use_locking, name);

        public void device(string device_name)
            => get_default_graph().device(device_name);

        public List<T> get_collection<T>(string key, string scope = "")
            => get_default_graph().get_collection<T>(key, scope: scope);

        /// <summary>
        /// A context manager that lifts ops out of control-flow scopes and function-building graphs.
        /// When eager execution is enabled, code inside an init_scope block runs with
        /// eager execution enabled even when tracing a `tf.function`. 
        /// </summary>
        public ops.NameScope init_scope()
            => ops.init_scope();

        /// <summary>
        /// Returns a context manager that creates hierarchical names for operations.
        /// </summary>
        /// <param name="name">The name argument that is passed to the op function.</param>
        /// <param name="default_name">The default name to use if the name argument is None.</param>
        /// <param name="values">The list of Tensor arguments that are passed to the op function.</param>
        /// <returns>The scope name.</returns>
        public ops.NameScope name_scope(string name, string default_name = "", object values = null)
            => new ops.NameScope(name, default_name, values);

        /// <summary>
        /// Does nothing. Only useful as a placeholder for control edges.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public Operation no_op(string name = null)
            => gen_control_flow_ops.no_op(name: name);

        /// <summary>
        /// map on the list of tensors unpacked from `elems` on dimension 0.
        /// </summary>
        /// <param name="fn"></param>
        /// <param name="elems"></param>
        /// <param name="dtype"></param>
        /// <param name="parallel_iterations"></param>
        /// <param name="back_prop"></param>
        /// <param name="swap_memory"></param>
        /// <param name="infer_shape"></param>
        /// <param name="name"></param>
        /// <returns>A tensor or (possibly nested) sequence of tensors.</returns>
        public Tensor map_fn(Func<Tensor, Tensor> fn,
            Tensor elems,
            TF_DataType dtype = TF_DataType.DtInvalid,
            int parallel_iterations = -1,
            bool back_prop = true,
            bool swap_memory = false,
            bool infer_shape = true,
            string name = null)
            => Operation.map_fn(fn,
                elems,
                dtype,
                parallel_iterations: parallel_iterations,
                back_prop: back_prop,
                swap_memory: swap_memory,
                infer_shape: infer_shape,
                name: name);
    }
}
