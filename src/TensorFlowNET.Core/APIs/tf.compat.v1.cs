/*****************************************************************************
   Copyright 2020 The TensorFlow.NET Authors. All Rights Reserved.

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
    public class CompatV1Api
    {
        public void disable_eager_execution()
            => tf.Context.graph_mode();

        public IVariableV1 get_variable(string name,
            Shape shape = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            object initializer = null, // IInitializer or Tensor
            bool? trainable = null,
            List<string> collections = null,
            bool? use_resource = null,
            bool validate_shape = true,
            VariableSynchronization synchronization = VariableSynchronization.Auto,
            VariableAggregation aggregation = VariableAggregation.None)
        {
            var scope = Tensorflow.variable_scope.get_variable_scope();
            var store = Tensorflow.variable_scope._get_default_variable_store();
            return scope.get_variable(store,
                name,
                shape: shape,
                dtype: dtype,
                use_resource: use_resource,
                validate_shape: validate_shape,
                initializer: initializer,
                trainable: trainable,
                collections: collections);
        }

        public Operation global_variables_initializer()
        {
            var g = variables.global_variables();
            return variables.variables_initializer(g.ToArray());
        }

        public Session Session()
            => new Session().as_default();
    }
}
