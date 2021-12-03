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

using System.Collections.Generic;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public IVariableV1[] global_variables(string scope = null)
        {
            return (ops.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope) as List<IVariableV1>)
                .ToArray();
        }

        /// <summary>
        /// Returns an Op that initializes a list of variables.
        /// </summary>
        /// <param name="var_list">List of `Variable` objects to initialize.</param>
        /// <param name="name">Optional name for the returned operation.</param>
        /// <returns>An Op that run the initializers of all the specified variables.</returns>
        public Operation variables_initializer(IVariableV1[] var_list, string name = "init")
            => variables.variables_initializer(var_list, name: name);

        public Operation global_variables_initializer()
            => tf.compat.v1.global_variables_initializer();

        /// <summary>
        /// Returns all variables created with `trainable=True`.
        /// </summary>
        /// <param name="scope"></param>
        /// <returns></returns>
        public IVariableV1[] trainable_variables(string scope = null)
            => (variables.trainable_variables() as List<IVariableV1>).ToArray();

        public VariableScope get_variable_scope()
            => Tensorflow.variable_scope.get_variable_scope();
    }
}
