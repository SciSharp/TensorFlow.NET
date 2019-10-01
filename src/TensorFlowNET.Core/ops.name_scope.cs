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
    public partial class ops
    {
        public static NameScope name_scope(string name,
            string default_name = "",
            object values = null) => new NameScope(name, default_name, values);

        /// <summary>
        /// Returns a context manager that creates hierarchical names for operations.
        /// </summary>
        public class NameScope : IObjectLife
        {
            public string _name;
            public string _default_name;
            public object _values;
            public string _name_scope;
            public string old_stack = "";
            
            public NameScope(string name, string default_name = "", object values = null)
            {
                _name = name;
                _default_name = default_name;
                _values = values;
            }

            public void __enter__()
            {
                _name = _name ?? _default_name;

                Graph g = null;

                if (_values is List<Tensor> vList)
                    g = _get_graph_from_inputs(vList.ToArray());
                else if (_values is Tensor[] vArray)
                    g = _get_graph_from_inputs(vArray);

                if (g == null)
                    g = get_default_graph();

                old_stack = g._name_stack;
                _name_scope = g.name_scope(_name);
            }

            public void Dispose()
            {
                var g = get_default_graph();
                g._name_stack = old_stack;
            }

            public void __exit__()
            {
            }

            public void __init__()
            {
                
            }

            public void __del__()
            {
                
            }

            /// <summary>
            /// __enter__()
            /// </summary>
            /// <param name="ns"></param>
            public static implicit operator string(NameScope ns)
            {
                return ns._name_scope;
            }
        }
    }
}
