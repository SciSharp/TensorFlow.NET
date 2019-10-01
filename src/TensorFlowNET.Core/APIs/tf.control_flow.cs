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

namespace Tensorflow
{
    public partial class tensorflow
    {
        public Tensor cond(Tensor pred,
            Func<ITensorOrOperation> true_fn = null,
            Func<ITensorOrOperation> false_fn = null,
            bool strict = false,
            string name = null)
            => control_flow_ops.cond(pred, true_fn, false_fn, strict: strict, name: name);

        /// <summary>
        /// Create an op that groups multiple operations.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="inputs"></param>
        /// <param name="name"></param>
        /// <returns>An Operation that executes all its inputs.</returns>
        public Operation group<T>(T[] inputs, string name = null) where T : ITensorOrOperation
            => control_flow_ops.group(inputs, name: name);

        public Tensor while_loop(Func<Tensor, Tensor> cond, Func<Tensor, Tensor> body, Tensor[] loop_vars,
            TensorShape shape_invariants = null,
            int parallel_iterations = 10,
            bool back_prop = true,
            bool swap_memory = false,
            string name = null,
            int? maximum_iterations = null,
            bool return_same_structure = false)
            => control_flow_ops.while_loop(cond, body, loop_vars,
                shape_invariants: shape_invariants,
                parallel_iterations: parallel_iterations,
                back_prop: back_prop,
                swap_memory: swap_memory,
                name: name,
                maximum_iterations: maximum_iterations,
                return_same_structure: return_same_structure);

        public _ControlDependenciesController control_dependencies(ITensorOrOperation[] control_inputs) 
            => ops.control_dependencies(control_inputs);
    }
}
