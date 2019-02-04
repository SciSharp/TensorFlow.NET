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
        /// <param name="value">A `Tensor`.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A `Tensor`. Has the same type as `value`.</returns>
        public static Tensor fill(Tensor dims, Tensor value, string name = "")
        {
            var _op = _op_def_lib._apply_op_helper("Fill", name, new { dims, value });

            return _op.outputs[0];
        }

        public static (Tensor, Tensor) broadcast_gradient_args(Tensor s0, Tensor s1, string name = "")
        {
            return (null, null);
        }
    }
}
