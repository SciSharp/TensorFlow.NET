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
using System.Linq;

namespace Tensorflow
{
    public class PureVariableScope : ITensorFlowObject
    {
        private string _name;
        private VariableScope _scope;
        private string _new_name;
        private string _old_name_scope;
        private bool _reuse;
        private _VariableStore _var_store;
        private VariableScope _old;
        private _VariableScopeStore _var_scope_store;
        private VariableScope variable_scope_object;
        private VariableScope _cached_variable_scope_object;
        VariableScope _last_variable_scope_object;
        Dictionary<string, int> _old_subscopes;
        public PureVariableScope(string name,
            string old_name_scope = null,
            TF_DataType dtype = TF_DataType.DtInvalid)
        {
            _name = name;
            _old_name_scope = old_name_scope;
            _var_store = variable_scope._get_default_variable_store();
            _var_scope_store = variable_scope.get_variable_scope_store();
        }

        public PureVariableScope(VariableScope scope,
            string old_name_scope = null,
            TF_DataType dtype = TF_DataType.DtInvalid)
        {
            _scope = scope;
            _old_name_scope = old_name_scope;
            _var_store = variable_scope._get_default_variable_store();
            _var_scope_store = variable_scope.get_variable_scope_store();
            _new_name = _scope.name;

            string name_scope = _scope._name_scope;
            variable_scope_object = new VariableScope(_reuse,
                name: _new_name,
                name_scope: name_scope);

            _cached_variable_scope_object = variable_scope_object;
        }

        public void __enter__()
        {
            _old = _var_scope_store.current_scope;
            if (_scope != null)
            {
                _var_scope_store.open_variable_scope(_new_name);
                _old_subscopes = _var_scope_store.variable_scopes_count.ToDictionary(kv => kv.Key, kv => kv.Value);
                variable_scope_object = _cached_variable_scope_object;
            }
            else
            {
                _new_name = string.IsNullOrEmpty(_old.name) ? _name : _old.name + "/" + _name;
                _reuse = _reuse || _old.resue;
                string name_scope = _old_name_scope == null ? _name : _old_name_scope;

                variable_scope_object = new VariableScope(_reuse,
                    name: _new_name,
                    name_scope: name_scope);

                _var_scope_store.open_variable_scope(_new_name);
            }
            _var_scope_store.current_scope = variable_scope_object;
            _last_variable_scope_object = variable_scope_object;
        }

        public void Dispose()
        {

        }

        public void __exit__()
        {
            // If jumping out from a non-prolonged scope, restore counts.
            if (_scope != null)
                _var_scope_store.variable_scopes_count = _old_subscopes;
            else
                _var_scope_store.close_variable_subscopes(_new_name);
            _var_scope_store.current_scope = _old;
        }

        public void __init__()
        {

        }

        public void __del__()
        {

        }

        public static implicit operator VariableScope(PureVariableScope scope)
        {
            return scope.variable_scope_object;
        }
    }
}
