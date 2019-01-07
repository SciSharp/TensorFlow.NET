using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Tensorflow
{
    public static class gen_math_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        public static Tensor add(Tensor x, Tensor y)
        {
            var keywords = new Dictionary<string, object>();
            keywords.Add("x", x);
            keywords.Add("y", y);

            var _op = _op_def_lib._apply_op_helper("Add", name: "add", keywords: keywords);

            return new Tensor(_op, 0, _op.OutputType(0));
        }

        public static Tensor sub(Tensor x, Tensor y)
        {
            var keywords = new Dictionary<string, object>();
            keywords.Add("x", x);
            keywords.Add("y", y);

            var _op = _op_def_lib._apply_op_helper("Sub", name: "sub", keywords: keywords);

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

        public static Tensor real_div(Tensor x, Tensor y)
        {
            var keywords = new Dictionary<string, object>();
            keywords.Add("x", x);
            keywords.Add("y", y);

            var _op = _op_def_lib._apply_op_helper("RealDiv", name: "truediv", keywords: keywords);

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

        public static Tensor pow(Tensor x, double y)
        {
            var keywords = new Dictionary<string, object>();
            keywords.Add("x", x);
            keywords.Add("y", y);

            var _op = _op_def_lib._apply_op_helper("Pow", name: "Pow", keywords: keywords);

            return new Tensor(_op, 0, _op.OutputType(0));
        }

        public static Tensor sum(Tensor input, int[] axis = null)
        {
            if(axis == null) axis = new int[0];
            var keywords = new Dictionary<string, object>();
            keywords.Add("input", input);
            keywords.Add("reduction_indices", constant_op.Constant(axis));
            keywords.Add("keep_dims", false);

            var _op = _op_def_lib._apply_op_helper("Sum", keywords: keywords);

            return new Tensor(_op, 0, _op.OutputType(0));
        }
    }
}
