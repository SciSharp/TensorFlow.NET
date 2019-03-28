using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class check_ops : Python
    {
        /// <summary>
        /// Assert the condition `x == y` holds element-wise.
        /// </summary>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <param name="name"></param>
        public static Operation assert_equal(object t1, object t2, object[] data = null, string name = null)
        {
            return with(ops.name_scope(name, "assert_equal", new { t1, t2, data }), delegate
            {
                var x = ops.convert_to_tensor(t1, name: "x");
                var y = ops.convert_to_tensor(t2, name: "y");
                var condition = math_ops.reduce_all(gen_math_ops.equal(x, y));
                var x_static = tensor_util.constant_value(x);
                var y_static = tensor_util.constant_value(y);
                return control_flow_ops.Asset(condition, data);
            });
        }
    }
}
