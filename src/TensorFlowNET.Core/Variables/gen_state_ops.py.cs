using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;

namespace Tensorflow
{
    public class gen_state_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();
        public static Execute _execute = new Execute();

        /// <summary>
        /// Holds state in the form of a tensor that persists across steps.
        /// Outputs a ref to the tensor state so it may be read or modified.
        /// </summary>
        /// <param name="shape">The shape of the variable tensor.</param>
        /// <param name="dtype">The type of elements in the variable tensor.</param>
        /// <param name="name"></param>
        /// <param name="container"></param>
        /// <param name="shared_name"></param>
        /// <returns></returns>
        public static Tensor variable_v2(long[] shape, TF_DataType dtype, string name = "", string container = "", string shared_name = "")
        {
            var keywords = new Dictionary<string, object>();
            keywords.Add("dtype", dtype);
            keywords.Add("shape", shape);
            keywords.Add("container", container);
            keywords.Add("shared_name", shared_name);

            var _op = _op_def_lib._apply_op_helper("VariableV2", name: name, keywords: keywords);

            var _result = _op.outputs;
            var _inputs_flat = _op.inputs;

            var _attrs = new Dictionary<string, object>();
            _attrs["dtype"] = _op.get_attr<DataType>("dtype");
            _attrs["shape"] = _op.get_attr<int[]>("shape");
            _attrs["container"] = _op.get_attr<string>("container");
            _attrs["shared_name"] = _op.get_attr<string>("shared_name");

            _execute.record_gradient("VariableV2", _inputs_flat, _attrs, _result, name);

            return _result[0];
        }

        /// <summary>
        /// Update 'ref' by assigning 'value' to it
        /// </summary>
        /// <param name="REF"></param>
        /// <param name="value"></param>
        /// <param name="validate_shape"></param>
        /// <param name="use_locking"></param>
        /// <param name="name"></param>
        public static Tensor assign(Tensor tensor, Tensor value, 
            bool validate_shape = true, 
            bool use_locking = true,
            string name = "")
        {
            var keywords = new Dictionary<string, object>();
            keywords.Add("ref", tensor);
            keywords.Add("value", value);
            keywords.Add("validate_shape", validate_shape);
            keywords.Add("use_locking", use_locking);

            var _op = _op_def_lib._apply_op_helper("Assign", name: name, keywords: keywords);

            var _result = _op.outputs;
            var _inputs_flat = _op.inputs;

            var _attrs = new Dictionary<string, object>();
            _attrs["T"] = _op.get_attr<DataType>("T");
            _attrs["validate_shape"] = _op.get_attr<bool>("validate_shape");
            _attrs["use_locking"] = _op.get_attr<bool>("use_locking");

            _execute.record_gradient("Assign", _inputs_flat, _attrs, _result, name);

            return _result[0];
        }
    }
}
