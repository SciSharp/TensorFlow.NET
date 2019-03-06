using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Operation global_variables_initializer()
        {
            var g = variables.global_variables();
            return variables.variables_initializer(g.ToArray());
        }

        public static RefVariable get_variable(string name,
            TensorShape shape = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            object initializer = null, // IInitializer or Tensor
            bool? trainable = null,
            VariableSynchronization synchronization = VariableSynchronization.AUTO,
            VariableAggregation aggregation = VariableAggregation.NONE)
        {
            var scope = Tensorflow.variable_scope.get_variable_scope();
            var store = Tensorflow.variable_scope._get_default_variable_store();
            return scope.get_variable(store, 
                name, 
                shape: shape, 
                dtype: dtype,
                initializer: initializer,
                trainable: trainable);
        }
    }
}
