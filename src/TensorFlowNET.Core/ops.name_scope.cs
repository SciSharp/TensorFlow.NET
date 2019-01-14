using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public partial class ops
    {
        public class name_scope<T> : IDisposable
        {
            public string _name;
            public string _default_name;
            public object _values;
            public Context _ctx;
            public string _name_scope;
            private object _g_manager;

            public name_scope(string name, string default_name = "", List<T> values = null)
            {
                _name = name;
                _default_name = default_name;
                _values = values;
                _ctx = new Context();
            }

            public string __enter__()
            {
                if (String.IsNullOrEmpty(_name))
                {
                    _name = _default_name;
                }

                Graph g = null;
                if (_values is List<Tensor> values)
                    g = _get_graph_from_inputs(values);

                if (g == null)
                    g = get_default_graph();

                _name_scope = g.name_scope(_name);

                return _name_scope;
            }

            public void Dispose()
            {
                var g = get_default_graph();
                g._name_stack = g.old_stack;
                // clear g._name_stack
                g.old_stack = "";
            }

            /// <summary>
            /// __enter__()
            /// </summary>
            /// <param name="ns"></param>
            public static implicit operator string(name_scope<T> ns)
            {
                if (string.IsNullOrEmpty(ns._name_scope))
                    return ns.__enter__();
                else
                    return ns._name_scope;
            }
        }
    }
}
