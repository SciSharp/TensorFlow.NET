using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public class variable_scope : IPython
    {
        public static string _VARSTORE_KEY = "__variable_store";
        public static string _VARSCOPESTORE_KEY = "__varscope";
        public static bool _DEFAULT_USE_RESOURCE = false;

        private bool _use_resource;
        public bool UseResource => _use_resource;
        private string _name;
        private VariableScope _scope;
        private string _default_name;
        private object _values;
        private ops.name_scope _current_name_scope;
        private bool _auxiliary_name_scope;
        private PureVariableScope _cached_pure_variable_scope;

        public variable_scope(string name, 
            string default_name = "", 
            object values = null,
            bool auxiliary_name_scope = true)
        {
            _name = name;
            _default_name = default_name;
            _values = values;
            _current_name_scope = null;

            _use_resource = false;
            if (_default_name == null && _name == null)
                throw new TypeError("If default_name is None then name is required");

            _auxiliary_name_scope = auxiliary_name_scope;
        }

        public variable_scope(VariableScope scope,
            string default_name = "",
            object values = null,
            bool auxiliary_name_scope = true)
        {
            _scope = scope;
            _default_name = default_name;
            _values = values;
            _current_name_scope = null;

            _use_resource = false;
            if (_default_name == null && _scope == null)
                throw new TypeError("If default_name is None then scope is required");

            _auxiliary_name_scope = auxiliary_name_scope;
        }

        public void __enter__()
        {
            _scope = _enter_scope_uncached();
        }

        private VariableScope _enter_scope_uncached()
        {
            ops.name_scope current_name_scope;
            if (_auxiliary_name_scope)
                // Create a new name scope later
                current_name_scope = null;
            else
            {
                // Reenter the current name scope
                string name_scope = ops.get_name_scope();
                if(!string.IsNullOrEmpty(name_scope))
                    // Hack to reenter
                    name_scope += "/";
                current_name_scope = new ops.name_scope(name_scope);
            }

            if (_name != null || _scope != null)
            {
                var name_scope = _name == null ? _scope._name.Split('/').Last() : _name;
                if (name_scope != null || current_name_scope != null)
                    current_name_scope = new ops.name_scope(name_scope);
                current_name_scope.__enter__();
                var current_name_scope_name = current_name_scope;
                _current_name_scope = current_name_scope;
                string old_name_scope = current_name_scope_name;
                PureVariableScope pure_variable_scope = null;
                if(_scope == null)
                    pure_variable_scope = new PureVariableScope(_name, old_name_scope: old_name_scope);
                else
                    pure_variable_scope = new PureVariableScope(_scope, old_name_scope: old_name_scope);
                pure_variable_scope.__enter__();
                VariableScope entered_pure_variable_scope = pure_variable_scope;
                _cached_pure_variable_scope = pure_variable_scope;
                return entered_pure_variable_scope;
            }

            throw new NotImplementedException("_enter_scope_uncached");
        }

        public static RefVariable default_variable_creator(object initial_value,
            string name = null,
            bool? trainable = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            bool validate_shape = false,
            bool ? use_resource = null, 
            VariableSynchronization synchronization = VariableSynchronization.AUTO,
            VariableAggregation aggregation = VariableAggregation.NONE)
        {
            trainable = _get_trainable_value(synchronization, trainable);
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

        public static bool _get_trainable_value(VariableSynchronization synchronization, bool? trainable = true)
        {
            if (synchronization == VariableSynchronization.ON_READ)
            {
                if (trainable.Value)
                    throw new ValueError("Synchronization value can be set to " +
                        "VariableSynchronization.ON_READ only for non-trainable variables. " +
                        "You have specified trainable=True and " +
                        "synchronization=VariableSynchronization.ON_READ.");
            }
            else if (!trainable.HasValue)
            {
                trainable = true;
            }
            
            return trainable.Value;
        }

        public static implicit operator VariableScope(variable_scope scope)
        {
            return scope._scope;
        }

        public void __exit__()
        {
            if (_current_name_scope != null)
                _current_name_scope.__exit__();
        }

        public void Dispose()
        {
            if (_current_name_scope != null)
                _current_name_scope.Dispose();
        }
    }
}
