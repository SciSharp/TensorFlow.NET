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

using Tensorflow.Contexts;
using static Tensorflow.Binding;

namespace Tensorflow.Eager
{
    /// <summary>
    /// python\eager\pywrap_tfe_src.cc
    /// </summary>
    public partial class EagerRunner
    {
        /// <summary>
        /// Execute a TensorFlow operation.
        /// </summary>
        /// <param name="op_name">
        /// Name of the TensorFlow operation (see REGISTER_OP in C++ code) to 
        /// execute.
        /// </param>
        /// <param name="num_outputs">
        /// The number of outputs of the operation to fetch.
        /// </param>
        /// <param name="inputs">
        /// A list of inputs to the operation. Each entry should be a Tensor, or
        /// a value which can be passed to the Tensor constructor to create one.
        /// </param>
        /// <param name="attrs">
        /// A tuple with alternating string attr names and attr values for this
        /// operation.
        /// </param>
        /// <param name="ctx">The value of context.context().</param>
        /// <param name="name">Customized name for the operation.</param>
        /// <returns>List of output Tensor objects. The list is empty if there are no outputs</returns>
        public Tensor[] Execute(Context ctx, string op_name, int num_outputs,
            Tensor[] inputs, object[] attrs,
            string name = null)
        {
            ctx.ensure_initialized();

            var results = tf.Runner.TFE_Execute(ctx,
               ctx.DeviceName,
               op_name,
               inputs,
               attrs,
               num_outputs);

            return results;
        }
    }
}