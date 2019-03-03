using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class PureVariableScope : IPython
    {
        private string _name_or_scope;
        private string _new_name;
        private string _old_name_scope;
        private bool _reuse;
        private _VariableStore _var_store;
        private VariableScope _old;
        private _VariableScopeStore _var_scope_store;
        private VariableScope variable_scope_object;

        public PureVariableScope(string name_or_scope, 
            string old_name_scope = null, 
            TF_DataType dtype = TF_DataType.DtInvalid)
        {
            _name_or_scope = name_or_scope;
            _old_name_scope = old_name_scope;
            _var_store = variable_scope._get_default_variable_store();
            _var_scope_store = variable_scope.get_variable_scope_store();
        }

        public void __enter__()
        {
            _old = _var_scope_store.current_scope;
            _new_name = string.IsNullOrEmpty(_old.name) ? _name_or_scope : _old.name + "/" + _name_or_scope;
            _reuse = _reuse || _old.resue;
            string name_scope = _old_name_scope == null ? _name_or_scope : _old_name_scope;
            
            variable_scope_object = new VariableScope(_reuse, 
                name: _new_name,
                name_scope: name_scope);

            _var_scope_store.open_variable_scope(_new_name);
            _var_scope_store.current_scope = variable_scope_object;
        }

        public void Dispose()
        {
            
        }

        public void __exit__()
        {
            
        }

        public static implicit operator VariableScope(PureVariableScope scope)
        {
            return scope.variable_scope_object;
        }
    }
}
