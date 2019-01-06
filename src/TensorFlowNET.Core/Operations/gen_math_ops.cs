using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Tensorflow
{
    public static class gen_math_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        public static Tensor add(Tensor a, Tensor b)
        {
            var keywords = new Dictionary<string, object>();
            keywords.Add("x", a);
            keywords.Add("y", b);

            var _op = _op_def_lib._apply_op_helper("Add", name: "add", keywords: keywords);

            return new Tensor(_op, 0, _op.OutputType(0));
        }
    }
}
