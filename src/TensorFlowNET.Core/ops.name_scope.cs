using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;

namespace Tensorflow
{
    public partial class ops
    {
        public class name_scope : IPython
        {
            public string _name;
            public string _default_name;
            public object _values;
            public Context _ctx;
            public string _name_scope;
            public string old_stack = "";
            private object _g_manager;

            public name_scope(string name, string default_name = "", object values = null)
            {
                _name = name;
                _default_name = default_name;
                _values = values;
                // _ctx = new Context();
            }

            public void __enter__()
            {
                _name = _name == null ? _default_name : _name;

                Graph g = null;
                if (_values is List<Tensor> values)
                    g = _get_graph_from_inputs(values);

                if (g == null)
                    g = get_default_graph();

                old_stack = g._name_stack;
                _name_scope = g.name_scope(_name);
            }

            public void Dispose()
            {
                var g = get_default_graph();
                // Console.WriteLine($"name_scope: {g._name_stack} -> {old_stack}");
                g._name_stack = old_stack;
            }

            public void __exit__()
            {
            }

            /// <summary>
            /// __enter__()
            /// </summary>
            /// <param name="ns"></param>
            public static implicit operator string(name_scope ns)
            {
                return ns._name_scope;
            }
        }
    }
}
