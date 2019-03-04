using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Variable scope object to carry defaults to provide to `get_variable`
    /// </summary>
    public class VariableScope : Python
    {
        public bool use_resource { get; set; }
        private _ReuseMode _reuse;
        public bool resue;

        private TF_DataType _dtype;
        public string name { get; set; }
        public string name_scope { get; set; }

        public VariableScope(bool reuse, 
            string name = "", 
            string name_scope = "",
            TF_DataType dtype = TF_DataType.TF_FLOAT)
        {
            this.name = name;
            this.name_scope = name_scope;
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
            return with(new ops.name_scope(null), scope =>
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
