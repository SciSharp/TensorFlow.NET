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
        public static Tensor variable_v2(long[] shape, TF_DataType dtype, string name = null, string container = "", string shared_name = "")
        {
            var _op = _op_def_lib._apply_op_helper("VariableV2", name: name, args: new { dtype, shape, container, shared_name });

            var _result = _op.outputs;
            var _inputs_flat = _op.inputs;

            var _attrs = new Dictionary<string, object>();
            _attrs["dtype"] = _op.get_attr("dtype");
            _attrs["shape"] = _op.get_attr("shape");
            _attrs["container"] = _op.get_attr("container");
            _attrs["shared_name"] = _op.get_attr("shared_name");

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
        public static Tensor assign(Tensor tensor, object value, 
            bool validate_shape = true, 
            bool use_locking = true,
            string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Assign", name: name, args: new { _ref_ = tensor, value, validate_shape, use_locking });

            var _result = _op.outputs;
            var _inputs_flat = _op.inputs;

            var _attrs = new Dictionary<string, object>();
            _attrs["T"] = _op.get_attr("T");
            _attrs["validate_shape"] = _op.get_attr("validate_shape");
            _attrs["use_locking"] = _op.get_attr("use_locking");

            _execute.record_gradient("Assign", _inputs_flat, _attrs, _result, name);

            return _result[0];
        }
    }
}
