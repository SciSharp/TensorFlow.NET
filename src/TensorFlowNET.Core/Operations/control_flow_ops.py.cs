using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Operations;
using util = Tensorflow.control_flow_util;

namespace Tensorflow
{
    public class control_flow_ops : Python
    {
        public static Operation Asset(Tensor condition, object[] data, int? summarize = null, string name = null)
        {
            return with(ops.name_scope(name, "Assert", new { condition, data }), scope =>
            {
                name = scope;
                var xs = ops.convert_n_to_tensor(data);
                condition = ops.convert_to_tensor(condition, name: "Condition");
                Func<Operation[]> true_assert = () => new Operation[]
                {
                    gen_logging_ops._assert(condition, data, summarize, name: "Assert")
                };

                Func<Operation[]> false_assert = () => new Operation[]
                {
                    gen_control_flow_ops.no_op()
                };

                var guarded_assert = cond(condition, false_assert, true_assert, name: "AssertGuard");

                return guarded_assert[0].op;
            });
        }

        public static Operation group<T>(T[] inputs, string name = null) where T : ITensorOrOperation
        {
            return with(ops.name_scope(name, "group_deps", inputs), scope =>
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

        private static Operation _GroupControlDeps(string dev, Operation[] deps, string name = null)
        {
            return with(ops.control_dependencies(deps), ctl =>
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

        public static Tensor[] tuple(Tensor[] tensors, string name = null, Operation[] control_inputs = null)
        {
            return with(ops.name_scope(name, "tuple", tensors), scope =>
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

        public static Tensor with_dependencies(Operation[] dependencies, Tensor output_tensor, string name = null)
        {
            var values = new List<object>();
            values.AddRange(dependencies);
            values.Add(output_tensor);

            return with(ops.name_scope(name, "control_dependency", values), scope =>
            {
                name = scope;

                return with(ops.control_dependencies(dependencies), ctl =>
                {
                    output_tensor = ops.convert_to_tensor_or_composite(output_tensor);
                    return _Identity(output_tensor, name: name);
                });
            });
        }

        public static Tensor _Identity(Tensor data, string name = null)
        {
            data = ops.internal_convert_to_tensor_or_composite(data, as_ref: true);
            if ((int)data.dtype > 100)
                throw new NotImplementedException("_Identity");
            else
                return gen_array_ops.identity(data, name: name);
        }

        /// <summary>
        /// Forwards `data` to an output determined by `pred`.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="pred"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static (Tensor, Tensor) _SwitchRefOrTensor(Tensor data, Tensor pred, string name = "Switch")
        {
            data = ops.convert_to_tensor_or_indexed_slices(data, name: "data");

            ops.colocate_with(data, ignore_existing: true);

            return @switch(data, pred, name: name);
        }

        public static Tensor[] cond<T>(Tensor pred,
            Func<T[]> true_fn = null,
            Func<T[]> false_fn = null,
            bool strict = false,
            string name = null)
        {
            return with(ops.name_scope(name, "cond", new { pred }), delegate
            {
                // Add the Switch to the graph.
                var (p_2, p_1) = @switch(pred, pred);
                var pivot_1 = array_ops.identity(p_1, name: "switch_t");
                var pivot_2 = array_ops.identity(p_2, name: "switch_f");
                pred = array_ops.identity(pred, name: "pred_id");

                // Disable the fetching of tensors that are only on one branch of cond.
                foreach (var tensor in new Tensor[] { p_1, p_2, pivot_1, pivot_2, pred })
                    tensor.op.graph.prevent_fetching(tensor.op);

                // Build the graph for the true branch in a new context.
                var context_t = new CondContext(pred, pivot_1, branch: 1);
                context_t.Enter();
                var (orig_res_t, res_t) = context_t.BuildCondBranch(true_fn);
                context_t.Exit();

                // Build the graph for the false branch in a new context.
                var context_f = new CondContext(pred, pivot_2, branch: 0);
                context_f.Enter();
                var (orig_res_f, res_f) = context_f.BuildCondBranch(false_fn);
                context_f.Exit();

                var res_t_flat = res_t;
                var res_f_flat = res_f;

                var merges = zip(res_f_flat, res_t_flat)
                    .Select(pair => merge(new Tensor[] { pair.Item1, pair.Item2 }))
                    .ToArray();

                merges = _convert_flows_to_tensorarrays(orig_res_t, merges);

                ops.add_to_collection(ops.GraphKeys.COND_CONTEXT, context_t);
                ops.add_to_collection(ops.GraphKeys.COND_CONTEXT, context_f);

                return merges;
            });
        }

        public static Tensor[] _convert_flows_to_tensorarrays<T>(T[] tensors_or_tensorarrays, Tensor[] tensors_or_flows)
        {
            // zip(tensors_or_tensorarrays, tensors_or_flows).Select((ta, t_or_flow) => ta).ToArray();
            return tensors_or_flows;
        }

        public static Tensor merge(Tensor[] inputs, string name = null)
        {
            return with(ops.name_scope(name, "Merge", inputs), scope =>
            {
                name = scope;
                inputs = inputs.Select(inp =>
                            ops.internal_convert_to_tensor_or_indexed_slices(inp, as_ref: true))
                        .ToArray();
                return gen_control_flow_ops.merge(inputs, name).Item1;
            });
        }

        /// <summary>
        /// Forwards `data` to an output determined by `pred`.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="pred"></param>
        /// <param name="dtype"></param>
        /// <param name="name"></param>
        public static (Tensor, Tensor) @switch(Tensor data, 
            Tensor pred, 
            TF_DataType dtype = TF_DataType.DtInvalid, 
            string name = null)
        {
            return with(ops.name_scope(name, "Switch", new { data, pred }), scope =>
            {
                name = scope;
                data = ops.internal_convert_to_tensor_or_indexed_slices(data,
                    dtype: dtype,
                    name: "data",
                    as_ref: true);

                pred = ops.convert_to_tensor(pred, name: "pred");

                return gen_control_flow_ops.@switch(data, pred, name: name);
            });
        }

        public static Tensor ZerosLikeOutsideLoop(Operation op, int index)
        {
            var val = op.outputs[index];
            if (!util.IsSwitch(op))
            {
                if (val.dtype == TF_DataType.TF_RESOURCE)
                    throw new NotImplementedException("ZerosLikeOutsideLoop");
                return array_ops.zeros_like(val, optimize: false);
            }

            throw new NotImplementedException("ZerosLikeOutsideLoop");
        }
    }
}
