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

        public static Tensor mul(Tensor x, Tensor y)
        {
            var keywords = new Dictionary<string, object>();
            keywords.Add("x", x);
            keywords.Add("y", y);

            var _op = _op_def_lib._apply_op_helper("Mul", name: "mul", keywords: keywords);

            return new Tensor(_op, 0, _op.OutputType(0));
        }

        public static Tensor mat_mul(Tensor a, Tensor b, bool transpose_a = false, bool transpose_b = false)
        {
            var keywords = new Dictionary<string, object>();
            keywords.Add("a", a);
            keywords.Add("b", b);
            keywords.Add("transpose_a", transpose_a);
            keywords.Add("transpose_b", transpose_b);

            var _op = _op_def_lib._apply_op_helper("MatMul", name: "MatMul", keywords: keywords);

            return new Tensor(_op, 0, _op.OutputType(0));
        }
    }
}
