using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class _VariableScopeStore
    {
        public VariableScope current_scope { get; set; }

        public _VariableScopeStore()
        {
            current_scope = new VariableScope();
        }
    }
}
