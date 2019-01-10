using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class gen_state_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

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

            return new Tensor(_op, 0, dtype);
        }
    }
}
