using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public class control_flow_ops
    {
        public static Operation group(List<Operation> inputs, string name = "")
        {
            using(var namescope = new ops.name_scope(name, "group_deps", inputs))
            {
                name = namescope;

                // Sorts *inputs according to their devices.
                var ops_on_device = new Dictionary<string, Operation[]>();
                foreach (var inp in inputs)
                {
                    ops_on_device[inp.Device] = new Operation[] { inp };
                }

                // 1-level tree. The root node is the returned NoOp node.
                if (ops_on_device.Count == 1)
                {
                    var dev = ops_on_device.Keys.First();
                    var deps = ops_on_device.Values.First();
                    return _GroupControlDeps(dev, deps, name);
                }

                // 2-level tree. The root node is the returned NoOp node.
                // deps contains 1 NoOp node for each device.
                return null;
            }
        }

        private static Operation _GroupControlDeps(string dev, Operation[] deps, string name = "")
        {
            Operation result = null;

            Python.with(ops.control_dependencies(deps), delegate
            {
                if (string.IsNullOrEmpty(dev))
                {
                    result = gen_control_flow_ops.no_op(name);
                }
                else
                {
                    result = gen_control_flow_ops.no_op(name);
                }
            });

            return result;
        }

        /// <summary>
        /// Create the state for all the while loops involved in one gradients().
        /// </summary>
        /// <param name="between_op_list"></param>
        /// <param name="between_ops"></param>
        /// <param name="colocate_gradients_with_ops"></param>
        public static object MaybeCreateControlFlowState(List<Operation> between_op_list, List<Operation> between_ops, bool colocate_gradients_with_ops)
        {
            object loop_state = null;

            foreach (var op in between_op_list)
            {
                if (IsLoopExit(op))
                {
                    if(loop_state == null)
                    {
                        // loop_state = ControlFlowState();
                    }
                }
            }

            return loop_state;
        }

        private static bool IsLoopExit(Operation op)
        {
            return op.OpType == "Exit" || op.OpType == "RefExit";
        }
    }
}
