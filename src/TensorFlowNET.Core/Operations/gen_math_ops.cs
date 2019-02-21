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

        public static Tensor add(Tensor x, Tensor y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Add", name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor cast(Tensor x, TF_DataType DstT, bool Truncate= false, string name= "")
        {
            var _op = _op_def_lib._apply_op_helper("Cast", name, args: new { x, DstT, Truncate });

            return _op.outputs[0];
        }

        public static Tensor neg(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Neg", name, args: new { x });

            return _op.outputs[0];
        }

        public static Tensor sub<Tx, Ty>(Tx x, Ty y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Sub", name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor mul(Tensor x, Tensor y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Mul", name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor real_div(Tensor x, Tensor y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("RealDiv", name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor floor_mod(Tensor x, Tensor y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("FloorMod", name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor floor_div(Tensor x, Tensor y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("FloorDiv", name, args: new { x, y });

            return _op.outputs[0];
        }

        /// <summary>
        /// Multiply the matrix "a" by the matrix "b".
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="transpose_a"></param>
        /// <param name="transpose_b"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor mat_mul(Tensor a, Tensor b, bool transpose_a = false, bool transpose_b = false, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("MatMul", name, args: new { a, b, transpose_a, transpose_b });

            return _op.outputs[0];
        }

        /// <summary>
        /// Returns the max of x and y (i.e. x > y ? x : y) element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor maximum<T1, T2>(T1 x, T2 y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Maximum", name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor pow<Tx, Ty>(Tx x, Ty y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Pow", name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor sum(Tensor input, Tensor axis = null, bool keep_dims = false, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Sum", name, args: new { input, reduction_indices = axis, keep_dims });

            return _op.outputs[0];
        }

        /// <summary>
        /// Creates a sequence of numbers.
        /// </summary>
        /// <param name="start"></param>
        /// <param name="limit"></param>
        /// <param name="delta"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor range(Tensor start, Tensor limit, Tensor delta, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Range", name, new { start, limit, delta });

            return _op.outputs[0];
        }
    }
}
