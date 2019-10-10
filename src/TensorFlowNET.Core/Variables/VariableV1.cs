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

namespace Tensorflow
{
    /// <summary>
    /// A variable maintains state in the graph across calls to `run()`. You add a
    /// variable to the graph by constructing an instance of the class `Variable`.
    /// 
    /// The `Variable()` constructor requires an initial value for the variable,
    /// which can be a `Tensor` of any type and shape. The initial value defines the
    /// type and shape of the variable. After construction, the type and shape of
    /// the variable are fixed. The value can be changed using one of the assign methods.
    /// https://tensorflow.org/guide/variables
    /// </summary>
    public class VariableV1
    {
        public virtual string name { get; }
        public virtual Tensor graph_element { get; }
        public virtual Operation op { get; }
        public virtual Operation initializer { get; }
        public Tensor _variable;
        public Graph graph => _variable.graph;
        public VariableV1(object initial_value = null,
            bool trainable = true,
            List<string> collections = null,
            bool validate_shape = true,
            string caching_device = "",
            string name = null,
            TF_DataType dtype = TF_DataType.DtInvalid)
        {

        }
    }
}
