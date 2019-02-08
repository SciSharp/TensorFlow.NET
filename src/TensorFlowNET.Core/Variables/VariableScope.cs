using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class VariableScope
    {
        public bool use_resource { get; set; }
        private _ReuseMode _reuse { get; set; }

        private object _regularizer;
        private TF_DataType _dtype;
        public string name { get; set; }

        public VariableScope(TF_DataType dtype = TF_DataType.TF_FLOAT)
        {
            _reuse = _ReuseMode.AUTO_REUSE;
            _dtype = dtype;
        }

        public RefVariable get_variable(_VariableStore var_store, 
            string name, 
            TensorShape shape = null, 
            TF_DataType dtype = TF_DataType.DtInvalid,
            IInitializer initializer = null,
            bool? trainable = null,
            VariableSynchronization synchronization = VariableSynchronization.AUTO,
            VariableAggregation aggregation= VariableAggregation.NONE)
        {
            string full_name = !string.IsNullOrEmpty(this.name) ? this.name + "/" + name : name;
            return Python.with<ops.name_scope, RefVariable>(new ops.name_scope(""), scope =>
            {
                if (dtype == TF_DataType.DtInvalid)
                    dtype = _dtype;

                return var_store.get_variable(full_name, 
                    shape: shape, 
                    dtype: dtype,
                    initializer: initializer,
                    trainable: trainable,
                    synchronization: synchronization,
                    aggregation: aggregation);
            });
        }
    }
}
