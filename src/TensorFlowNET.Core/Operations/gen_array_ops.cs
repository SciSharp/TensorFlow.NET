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
            var keywords = new Dictionary<string, object>();
            keywords.Add("dtype", dtype);
            keywords.Add("shape", shape);

            var _op = _op_def_lib._apply_op_helper("Placeholder", keywords: keywords);
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
            var keywords = new Dictionary<string, object>();
            keywords.Add("input", input);

            var _op = _op_def_lib._apply_op_helper("Identity", name, keywords);

            return _op.outputs[0];
        }

        public static Tensor rank(Tensor input, string name = "")
        {
            var keywords = new Dictionary<string, object>();
            keywords.Add("input", input);

            var _op = _op_def_lib._apply_op_helper("Rank", name: name, keywords: keywords);

            return _op.outputs[0];
        }

        /// <summary>
        /// Creates a tensor filled with a scalar value.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="dims"></param>
        /// <param name="value"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor fill<T>(int[] dims, T value, string name = "")
        {
            var keywords = new Dictionary<string, object>();
            keywords.Add("dims", dims);
            keywords.Add("value", value);

            var _op = _op_def_lib._apply_op_helper("Fill", name);

            return _op.outputs[0];
        }
    }
}
