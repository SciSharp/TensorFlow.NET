using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class control_flow_ops
    {
        public static Operation group(List<Operation> inputs, string name = "")
        {
            using(var namescope = new ops.name_scope<Operation>(name, "group_deps", inputs))
            {
                // Sorts *inputs according to their devices.

                return _GroupControlDeps("", name);
            }
        }

        private static Operation _GroupControlDeps(string dev, string name = "")
        {
            return null;
        }
    }
}
