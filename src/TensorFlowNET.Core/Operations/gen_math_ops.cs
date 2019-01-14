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

            var _op = _op_def_lib._apply_op_helper("Add", keywords: keywords);

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

            var _op = _op_def_lib._apply_op_helper("Mul", keywords: keywords);

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

            var _op = _op_def_lib._apply_op_helper("MatMul", keywords: keywords);

            return new Tensor(_op, 0, _op.OutputType(0));
        }

        public static Tensor pow(Tensor x, double y)
        {
            var keywords = new Dictionary<string, object>();
            keywords.Add("x", x);
            keywords.Add("y", y);

            var _op = _op_def_lib._apply_op_helper("Pow", keywords: keywords);

            return new Tensor(_op, 0, _op.OutputType(0));
        }

        public static Tensor sum(Tensor input, Tensor axis = null)
        {
            var keywords = new Dictionary<string, object>();
            keywords.Add("input", input);
            keywords.Add("reduction_indices", axis);
            keywords.Add("keep_dims", false);

            var _op = _op_def_lib._apply_op_helper("Sum", keywords: keywords);

            return new Tensor(_op, 0, _op.OutputType(0));
        }

        /// <summary>
        /// Creates a sequence of numbers.
        /// </summary>
        /// <param name="start"></param>
        /// <param name="limit"></param>
        /// <param name="delta"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor range(Tensor start, Tensor limit, Tensor delta, string name = "")
        {
            var keywords = new Dictionary<string, object>();
            keywords.Add("start", start);
            keywords.Add("limit", limit);
            keywords.Add("delta", delta);

            var _op = _op_def_lib._apply_op_helper("Range", name, keywords);

            return _op.outputs[0];
        }
    }
}
