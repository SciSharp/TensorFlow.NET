using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class variable_scope
    {
        public static string _VARSTORE_KEY = "__variable_store";
        public static string _VARSCOPESTORE_KEY = "__varscope";
        public static bool _DEFAULT_USE_RESOURCE = false;

        public static RefVariable default_variable_creator(object initial_value, string name = "", TF_DataType dtype = TF_DataType.DtInvalid, bool ? use_resource = null, VariableSynchronization synchronization = VariableSynchronization.AUTO)
        {
            var trainable = _get_trainable_value(synchronization);
            if (!use_resource.HasValue)
            {
                use_resource = get_variable_scope().use_resource;
            }

            if(!use_resource.HasValue)
                use_resource = _DEFAULT_USE_RESOURCE;

            if (use_resource.Value)
            {
                throw new NotImplementedException();
            }
            else
            {
                return new RefVariable(initial_value, 
                    name: name,
                    dtype: dtype);
            }
        }

        public static _VariableStore _get_default_variable_store()
        {
            var store = ops.get_collection(_VARSTORE_KEY);
            if (store != null)
                return (store as List<_VariableStore>)[0];

            var store1 = new _VariableStore();
            ops.add_to_collection(_VARSTORE_KEY, store1);
            return store1;
        }

        public static VariableScope get_variable_scope()
        {
            return get_variable_scope_store().current_scope;
        }

        public static _VariableScopeStore get_variable_scope_store()
        {
            _VariableScopeStore ret = null;
            var scope_store = ops.get_collection(_VARSCOPESTORE_KEY);
            if (scope_store == null)
            {
                ret = new _VariableScopeStore();
                ops.add_to_collection(_VARSCOPESTORE_KEY, ret);
            }
            else
            {
                switch (scope_store)
                {
                    case List<RefVariable> values:
                        ret = values[0];
                        break;
                    case List<_VariableScopeStore> values:
                        ret = values[0];
                        break;
                    default:
                        throw new InvalidOperationException("get_variable_scope_store");
                }
                
            }

            return ret;
        }

        public static bool _get_trainable_value(VariableSynchronization synchronization, bool trainable = true)
        {
            if (synchronization == VariableSynchronization.ON_READ)
            {
                if (trainable)
                    throw new ValueError("Synchronization value can be set to " +
                        "VariableSynchronization.ON_READ only for non-trainable variables. " +
                        "You have specified trainable=True and " +
                        "synchronization=VariableSynchronization.ON_READ.");
            }
            
            return trainable;
        }
    }
}
