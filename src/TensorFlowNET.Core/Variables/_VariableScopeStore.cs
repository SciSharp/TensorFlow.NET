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

namespace Tensorflow
{
    public class _VariableScopeStore
    {
        public VariableScope current_scope { get; set; }
        public Dictionary<string, int> variable_scopes_count;

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

        public void close_variable_subscopes(string scope_name)
        {
            var variable_scopes_count_tmp = new Dictionary<string, int>();
            foreach (var k in variable_scopes_count.Keys)
                variable_scopes_count_tmp.Add(k, variable_scopes_count[k]);

            foreach (var k in variable_scopes_count_tmp.Keys)
                if (scope_name == null || k.StartsWith(scope_name + "/"))
                    variable_scopes_count[k] = 0;
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
