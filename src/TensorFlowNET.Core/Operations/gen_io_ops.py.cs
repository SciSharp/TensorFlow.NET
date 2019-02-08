using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class gen_io_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        public static Operation save_v2(Tensor prefix, string[] tensor_names, string[] shape_and_slices, Tensor[] tensors, string name = "")
        {
            var _op = _op_def_lib._apply_op_helper("SaveV2", name: name, args: new { prefix, tensor_names, shape_and_slices, tensors });

            return _op;
        }
    }
}
