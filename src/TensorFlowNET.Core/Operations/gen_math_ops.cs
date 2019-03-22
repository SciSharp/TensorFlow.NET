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
        /// <summary>
        /// Computes the mean of elements across dimensions of a tensor.
        /// Reduces `input` along the dimensions given in `axis`. Unless        /// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in        /// `axis`. If `keep_dims` is true, the reduced dimensions are retained with length 1.
        /// </summary>
        /// <param name="input">A `Tensor`. Must be one of the following types: 
        /// `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`. 
        /// The tensor to reduce.</param>
        /// <param name="axis">A `Tensor`. Must be one of the following types: `int32`, `int64`. The dimensions to reduce.</param>
        /// <param name="keep_dims"> An optional `bool`. Defaults to `False`. If true, retain reduced dimensions with length 1.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns> A `Tensor`. Has the same type as `input`.</returns>
        public static Tensor mean<T1, T2>(T1 input, T2 axis, bool keep_dims= false, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Mean", name, args: new { input, reduction_indices = axis, keep_dims = keep_dims });

            return _op.outputs[0];
        }

        public static Tensor prod<T1, T2>(T1 input, T2 axis, bool keep_dims = false, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Prod", name, args: new { input, reduction_indices = axis, keep_dims });

            return _op.outputs[0];
        }

        public static Tensor add(Tensor x, Tensor y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Add", name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor squared_difference(Tensor x, Tensor y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("SquaredDifference", name, args: new { x, y, name });

            return _op.outputs[0];
        }

        /// <summary>
        /// Computes square of x element-wise.
        /// </summary>
        /// <param name="x"> A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns> A `Tensor`. Has the same type as `x`.</returns>
        public static Tensor square(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Square", name, args: new { x });

            return _op.outputs[0];
        }

        /// <summary>
        /// Returns which elements of x are finite.
        /// </summary>
        /// <param name="x"> A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns> A `Tensor` of type `bool`.</returns>
        public static Tensor is_finite(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("IsFinite", name, args: new { x });

            return _op.outputs[0];
        }

        /// <summary>
        /// Computes exponential of x element-wise.  \\(y = e^x\\).
        /// </summary>
        /// <param name="x"> A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns> A `Tensor`. Has the same type as `x`.</returns>
        public static Tensor exp(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Exp", name, args: new { x });

            return _op.outputs[0];
        }

        /// <summary>
        /// Computes natural logarithm of x element-wise.
        /// </summary>
        /// <param name="x"> A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.</param>
        /// <param name="name"> name: A name for the operation (optional).</param>
        /// <returns> A `Tensor`. Has the same type as `x`.</returns>
        public static Tensor log(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Log", name, args: new { x });

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

        public static Tensor sqrt(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Sqrt", name, args: new { x });

            return _op.outputs[0];
        }

        public static Tensor sub<Tx, Ty>(Tx x, Ty y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Sub", name, args: new { x, y });

            return _op.outputs[0];
        }

        /// <summary>
        /// Returns the truth value of (x == y) element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor equal(Tensor x, Tensor y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Equal", name, args: new { x, y });

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

        public static Tensor reciprocal(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Reciprocal", name, args: new { x });

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

        public static Tensor _max<Tx, Ty>(Tx input, Ty axis, bool keep_dims=false, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Max", name, new { input, reduction_indices = axis, keep_dims });

            return _op.outputs[0];
        }

        public static Tensor pow<Tx, Ty>(Tx x, Ty y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Pow", name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor _sum(Tensor input, Tensor axis = null, bool keep_dims = false, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Sum", name, args: new { input, reduction_indices = axis, keep_dims });

            return _op.outputs[0];
        }

        public static Tensor _sum(Tensor input, int axis, bool keep_dims = false, string name = null)
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

        /// <summary>
        /// Returns the index with the largest value across dimensions of a tensor.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="dimension"></param>
        /// <param name="output_type"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor arg_max(Tensor input, int dimension, TF_DataType output_type = TF_DataType.TF_INT64, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("ArgMax", name, new { input, dimension, output_type });

            return _op.outputs[0];
        }
    }
}
