using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public partial class ops
    {
        public class name_scope
        {
            public string _name;
            public string _default_name;
            public object _values;
            public Context _ctx;
            public string _name_scope;

            public name_scope(string name, string default_name, List<object> values)
            {
                _name = name;
                _default_name = default_name;
                _values = values;
                _ctx = new Context();

                _name_scope = __enter__();
            }

            public string __enter__()
            {
                if (String.IsNullOrEmpty(_name))
                {
                    _name = _default_name;
                }

                var g = get_default_graph();
                return g.name_scope(_name);
            }

            public static implicit operator string(name_scope ns)
            {
                return ns._name_scope;
            }
        }
    }
}
