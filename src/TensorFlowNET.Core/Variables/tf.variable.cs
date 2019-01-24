using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Operation global_variables_initializer()
        {
            var g = variables.global_variables();
            return variables.variables_initializer(g.ToArray());
        }
    }
}
