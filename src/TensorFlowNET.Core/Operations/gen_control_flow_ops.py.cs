using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class gen_control_flow_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        public static Operation no_op(string name = "")
        {
            var _op = _op_def_lib._apply_op_helper("NoOp", name);

            return _op;
        }
    }
}
