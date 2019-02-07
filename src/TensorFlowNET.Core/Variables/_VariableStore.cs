using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Variable store that carries a number of named Variables.
    /// </summary>
    public class _VariableStore
    {
        private Dictionary<string, object> _vars;
        private Dictionary<string, object> _partitioned_vars;
        private bool _store_eager_variables;

        public _VariableStore()
        {
            _vars = new Dictionary<string, object>();
            _partitioned_vars = new Dictionary<string, object>();
            _store_eager_variables = false;
        }

        public RefVariable get_variable(string name,
            TensorShape shape = null,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            IInitializer initializer = null,
            bool trainable = false,
            bool validate_shape = true,
            VariableSynchronization synchronization = VariableSynchronization.AUTO,
            VariableAggregation aggregation = VariableAggregation.NONE)
        {
            dtype = dtype.as_base_dtype();
            trainable = variable_scope._get_trainable_value(synchronization, trainable);

            return _true_getter(name, 
                shape: shape, 
                dtype: dtype, 
                initializer: initializer,
                trainable: trainable,
                validate_shape: validate_shape,
                synchronization: synchronization,
                aggregation: aggregation);
        }

        private RefVariable _true_getter(string name,
            TensorShape shape = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            IInitializer initializer = null,
            bool trainable = false,
            bool validate_shape = true,
            VariableSynchronization synchronization = VariableSynchronization.AUTO,
            VariableAggregation aggregation = VariableAggregation.NONE)
        {
            return _get_single_variable(name: name);
        }

        private RefVariable _get_single_variable(string name,
            TensorShape shape = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            IInitializer initializer = null,
            bool reuse = false,
            bool trainable = false,
            bool validate_shape = false,
            VariableSynchronization synchronization = VariableSynchronization.AUTO,
            VariableAggregation aggregation = VariableAggregation.NONE)
        {
            if (_vars.ContainsKey(name))
            {
                if (!reuse)
                {
                    var var = _vars[name];

                }
                throw new NotImplementedException("_get_single_variable");
            }

            throw new NotImplementedException("_get_single_variable");
        }
    }
}
