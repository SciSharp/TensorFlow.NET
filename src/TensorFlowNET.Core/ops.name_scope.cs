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
using System.Diagnostics;
using Tensorflow.Contexts;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class ops
    {
        public static NameScope name_scope(string name,
            string default_name = "",
            object values = null,
            bool skip_on_eager = true) => new NameScope(name, default_name, values: values, skip_on_eager: skip_on_eager);

        /// <summary>
        /// Returns a context manager that creates hierarchical names for operations.
        /// </summary>
        public class NameScope : ITensorFlowObject
        {
            public string _name;
            public string _default_name;
            public object _values;
            public string scope_name;
            public string old_scope_name = "";
            bool _skip_on_eager = false;

            public NameScope(string name, string default_name = "", object values = null, bool skip_on_eager = true)
            {
                _name = name;
                _default_name = default_name;
                _values = values;
                _skip_on_eager = skip_on_eager;
            }

            [DebuggerStepThrough]
            public void __enter__()
            {
                if (tf.Context.executing_eagerly())
                {
                    (scope_name, old_scope_name) = enter_eager_name_scope(tf.Context, _name);
                }
                else
                {
                    _name = _name ?? _default_name;
                    Graph g = null;

                    if (_values is List<Tensor> vList)
                        g = _get_graph_from_inputs(vList.ToArray());
                    else if (_values is Tensor[] vArray)
                        g = _get_graph_from_inputs(vArray);

                    if (g == null)
                        g = get_default_graph();

                    old_scope_name = g._name_stack;
                    scope_name = g.name_scope(_name);
                }
            }

            private (string, string) enter_eager_name_scope(Context ctx, string name)
            {
                if (_skip_on_eager)
                    return (null, null);

                if (name == null)
                    name = _default_name;

                var scope_name = name;
                var old_name = ctx.ScopeName;
                // A trailing slash breaks out of nested name scopes, indicating a
                // fully specified scope name, for compatibility with Graph.name_scope.
                if (!name.EndsWith("/"))
                {
                    scope_name = name + "/";
                    if (!string.IsNullOrEmpty(old_name))
                        scope_name = old_name + scope_name;
                }

                ctx.ScopeName = scope_name;
                return (scope_name, old_name);
            }

            [DebuggerStepThrough]
            public void Dispose()
            {
                if (tf.Context.executing_eagerly())
                    tf.Context.ScopeName = old_scope_name;
                else
                    get_default_graph()._name_stack = old_scope_name;
            }

            [DebuggerStepThrough]
            public void __exit__()
            {
            }

            [DebuggerNonUserCode]
            public void __init__()
            {

            }

            [DebuggerNonUserCode]
            public void __del__()
            {

            }

            /// <summary>
            /// __enter__()
            /// </summary>
            /// <param name="ns"></param>
            public static implicit operator string(NameScope ns)
            {
                return ns.scope_name;
            }
        }
    }
}
