using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class gen_logging_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        public static Operation _assert(Tensor condition, object[] data, int? summarize = 3, string name = null)
        {
            if (!summarize.HasValue)
                summarize = 3;

            var _op = _op_def_lib._apply_op_helper("Assert", name, args: new { condition, data, summarize });

            return _op;
        }
    }
}
