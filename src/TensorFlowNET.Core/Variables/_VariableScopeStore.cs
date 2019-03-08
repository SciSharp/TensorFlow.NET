using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class _VariableScopeStore
    {
        public VariableScope current_scope { get; set; }
        private Dictionary<string, int> variable_scopes_count;

        public _VariableScopeStore()
        {
            current_scope = new VariableScope(false);
            variable_scopes_count = new Dictionary<string, int>();
        }

        public void open_variable_scope(string scope_name)
        {
            if (variable_scopes_count.ContainsKey(scope_name))
                variable_scopes_count[scope_name] += 1;
            else
                variable_scopes_count[scope_name] = 1;
        }

        public int variable_scope_count(string scope_name)
        {
            if (variable_scopes_count.ContainsKey(scope_name))
                return variable_scopes_count[scope_name];
            else
                return 0;
        }
    }
}
