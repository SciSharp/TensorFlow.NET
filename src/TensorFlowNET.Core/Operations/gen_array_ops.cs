using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;
using Tensorflow.Eager;

namespace Tensorflow
{
    public static class gen_array_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();
        public static Execute _execute = new Execute();

        public static Tensor placeholder(TF_DataType dtype, TensorShape shape = null, string name = "")
        {
            var _op = _op_def_lib._apply_op_helper("Placeholder", args: new { dtype, shape });
            var _result = _op.outputs;
            var _inputs_flat = _op.inputs;

            var _attrs = new Dictionary<string, object>();
            _attrs["dtype"] = _op.get_attr<DataType>("dtype");
            _attrs["shape"] = _op.get_attr<int[]>("shape");

            _execute.record_gradient("Placeholder", _inputs_flat, _attrs, _result, name);

            return new Tensor(_op, 0, dtype);
        }

        /// <summary>
        /// Return a tensor with the same shape and contents as the input tensor or value.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="name"></param>
        public static Tensor identity(Tensor input, string name = "")
        {
            var _op = _op_def_lib._apply_op_helper("Identity", name, new { input });

            return _op.outputs[0];
        }

        public static Tensor rank(Tensor input, string name = "")
        {
            var _op = _op_def_lib._apply_op_helper("Rank", name: name, args: new { input });

            return _op.outputs[0];
        }

        /// <summary>
        /// Creates a tensor filled with a scalar value.
        /// </summary>
        /// <param name="dims">A `Tensor`.</param>
        /// <param name="value">A `Tensor`. 0-D (scalar). Value to fill the returned tensor.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A `Tensor`. Has the same type as `value`.</returns>
        public static Tensor fill<T>(Tensor dims, T value, string name = "")
        {
            var _op = _op_def_lib._apply_op_helper("Fill", name, new { dims, value });

            return _op.outputs[0];
        }

        /// <summary>
        /// Return the reduction indices for computing gradients of s0 op s1 with broadcast.
        /// </summary>
        /// <param name="s0">A `Tensor`. Must be one of the following types: `int32`, `int64`.</param>
        /// <param name="s1">A `Tensor`. Must have the same type as `s0`.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A tuple of `Tensor` objects (r0, r1).</returns>
        public static (Tensor, Tensor) broadcast_gradient_args(Tensor s0, Tensor s1, string name = "")
        {
            var _op = _op_def_lib._apply_op_helper("BroadcastGradientArgs", name, new { s0, s1 });

            return (_op.outputs[0], _op.outputs[1]);
        }

        public static Tensor reshape(Tensor tensor, Tensor shape, string name = "")
        {
            var _op = _op_def_lib._apply_op_helper("Reshape", name, new { tensor, shape });
            return _op.outputs[0];
        }

        public static Tensor shape(Tensor input, TF_DataType out_type = TF_DataType.TF_INT32, string name = "")
        {
            var _op = _op_def_lib._apply_op_helper("Shape", name, new { input, out_type });
            return _op.outputs[0];
        }

        public static Tensor size(Tensor input, TF_DataType out_type = TF_DataType.TF_INT32, string name = "")
        {
            var _op = _op_def_lib._apply_op_helper("Size", name, new { input, out_type });
            return _op.outputs[0];
        }
    }
}
