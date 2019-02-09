using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public class control_flow_ops
    {
        public static Operation group<T>(T[] inputs, string name = "") where T : ITensorOrOperation
        {
            return Python.with<ops.name_scope, Operation>(new ops.name_scope(name, "group_deps", inputs), scope =>
            {
                name = scope;

                // Sorts *inputs according to their devices.
                var ops_on_device = new Dictionary<string, List<T>>();
                foreach (var inp in inputs)
                {
                    if (ops_on_device.ContainsKey(inp.Device))
                        ops_on_device[inp.Device].Add(inp);
                    else
                        ops_on_device[inp.Device] = new List<T> { inp };
                }

                // 1-level tree. The root node is the returned NoOp node.
                if (ops_on_device.Count == 1)
                {
                    var dev = ops_on_device.Keys.First();
                    var deps = ops_on_device.Values.First();
                    return _GroupControlDeps(dev, deps.Select(x => x.op).ToArray(), name);
                }

                // 2-level tree. The root node is the returned NoOp node.
                // deps contains 1 NoOp node for each device.
                throw new NotImplementedException("control_flow_ops.group");
            });
        }

        private static Operation _GroupControlDeps(string dev, Operation[] deps, string name = "")
        {
            return Python.with<_ControlDependenciesController, Operation>(ops.control_dependencies(deps), ctl =>
            {
                if (dev == null)
                {
                    return gen_control_flow_ops.no_op(name);
                }
                else
                {
                    return gen_control_flow_ops.no_op(name);
                }
            });
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

        public static Tensor[] tuple(Tensor[] tensors, string name = "", Operation[] control_inputs = null)
        {
            return Python.with<ops.name_scope, Tensor[]>(new ops.name_scope(name, "tuple", tensors), scope =>
            {
                name = scope;
                var gating_ops = tensors.Select(x => x.op).ToList();

                if(control_inputs != null)
                {
                    foreach (var c in control_inputs)
                        gating_ops.Add(c);
                }

                // Note that in order to ensure ordering in the pbtxt, we must take care to
                // ensure the order here.
                gating_ops = gating_ops.OrderBy(x => x._id).ToList();
                var gate = group(gating_ops.ToArray());

                var tpl = new List<Tensor>();
                foreach(var t in tensors)
                {
                    tpl.Add(with_dependencies(new Operation[] { gate }, t));
                }

                return tpl.ToArray();
            });
        }

        public static Tensor with_dependencies(Operation[] dependencies, Tensor output_tensor, string name = "")
        {
            var values = new List<object>();
            values.AddRange(dependencies);
            values.Add(output_tensor);

            return Python.with<ops.name_scope, Tensor>(new ops.name_scope(name, "control_dependency", values), scope =>
            {
                name = scope;

                return Python.with<_ControlDependenciesController, Tensor>(ops.control_dependencies(dependencies), ctl =>
                {
                    output_tensor = ops.convert_to_tensor_or_composite(output_tensor);
                    return _Identity(output_tensor, name: name);
                });
            });
        }

        public static Tensor _Identity(Tensor data, string name = "")
        {
            data = ops.internal_convert_to_tensor_or_composite(data, as_ref: true);
            if ((int)data.dtype > 100)
                throw new NotImplementedException("_Identity");
            else
                return gen_array_ops.identity(data, name: name);
        }
    }
}
