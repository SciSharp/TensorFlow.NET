using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class variable_scope
    {
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

        public static VariableScope get_variable_scope()
        {
            return get_variable_scope_store().current_scope;
        }

        public static _VariableScopeStore get_variable_scope_store()
        {
            var scope_store = ops.get_collection(_VARSCOPESTORE_KEY);
            if (scope_store == null)
            {
                scope_store = new _VariableScopeStore();
                ops.add_to_collection(_VARSCOPESTORE_KEY, scope_store);
            }
            else
            {
                // scope_store = scope_store[0];
            }

            return scope_store;
        }

        public static bool _get_trainable_value(VariableSynchronization synchronization, bool? trainable = null)
        {
            if(synchronization == VariableSynchronization.ON_READ)
            {
                if (trainable.Value)
                    throw new ValueError("Synchronization value can be set to " +
                        "VariableSynchronization.ON_READ only for non-trainable variables. " +
                        "You have specified trainable=True and " +
                        "synchronization=VariableSynchronization.ON_READ.");
                else
                    trainable = false;
            }
            else if (!trainable.HasValue)
            {
                trainable = true;
            }

            return trainable.Value;
        }
    }
}
