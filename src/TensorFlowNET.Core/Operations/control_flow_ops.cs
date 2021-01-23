/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Operations;
using Tensorflow.Operations.ControlFlows;
using Tensorflow.Util;
using static Tensorflow.Binding;
using util = Tensorflow.control_flow_util;

namespace Tensorflow
{
    public class control_flow_ops
    {
        public static Tensor _AddNextAndBackEdge(Tensor m, Tensor v, bool enforce_shape_invariant = true)
        {
            v = ops.convert_to_tensor(v);
            v = _NextIteration(v);
            if (enforce_shape_invariant)
                _EnforceShapeInvariant(m, v);
            m.op._update_input(1, v);
            return v;
        }

        /// <summary>
        /// Check if the shapes of the loops variables are invariants.
        /// </summary>
        /// <param name="merge_var"></param>
        /// <param name="next_var"></param>
        public static void _EnforceShapeInvariant(Tensor merge_var, Tensor next_var)
        {

        }

        public static Tensor exit(Tensor data, string name = null)
        {
            data = ops.internal_convert_to_tensor_or_indexed_slices(data, as_ref: true);
            if (data.dtype.is_ref_dtype())
                return gen_control_flow_ops.ref_exit(data, name: name);
            else
                return gen_control_flow_ops._exit(data, name: name);
        }

        public static Tensor _NextIteration(Tensor data, string name = null)
        {
            data = ops.internal_convert_to_tensor_or_indexed_slices(data, as_ref: true);
            if (data.dtype.is_ref_dtype())
                return gen_control_flow_ops.ref_next_iteration(data, name: name);
            else
                return gen_control_flow_ops.next_iteration(data, name: name);
        }

        public static Operation Assert(Tensor condition, object[] data, long summarize = 3, string name = null)
        {
            if (tf.executing_eagerly())
            {
                if (condition == null)
                    throw new InvalidArgumentError("");

                return null;
            }

            return tf_with(ops.name_scope(name, "Assert", new { condition, data }), scope =>
            {
                name = scope;
                var xs = ops.convert_n_to_tensor(data);
                condition = ops.convert_to_tensor(condition, name: "Condition");
                Func<Operation[]> true_assert = () =>
                {
                    var assert = gen_logging_ops.assert(condition, data, summarize, name: "Assert");
                    return new Operation[] { assert };
                };

                Func<Operation[]> false_assert = () =>
                {
                    var op = gen_control_flow_ops.no_op();
                    return new Operation[] { op };
                };

                var guarded_assert = cond(condition, false_assert, true_assert, name: "AssertGuard");

                return guarded_assert[0].op;
            });
        }

        public static Operation group<T>(T[] inputs, string name = null) where T : ITensorOrOperation
        {
            return tf_with(ops.name_scope(name, "group_deps", inputs), scope =>
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

        /// <summary>
        /// Does nothing. Only useful as a placeholder for control edges.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Operation no_op(string name = null)
            => gen_control_flow_ops.no_op(name: name);

        private static Operation _GroupControlDeps(string dev, Operation[] deps, string name = null)
        {
            return tf_with(ops.control_dependencies(deps), ctl =>
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
        public static ControlFlowState MaybeCreateControlFlowState(List<Operation> between_op_list, List<Operation> between_ops, bool colocate_gradients_with_ops)
        {
            var flag = new List<Operation>();
            ControlFlowState loop_state = null;

            int pos = 0;
            while (pos < between_op_list.Count)
            {
                var op = between_op_list[pos];
                if (IsLoopExit(op))
                {
                    if (loop_state == null)
                    {
                        loop_state = new ControlFlowState();
                    }
                    if (colocate_gradients_with_ops)
                        ops.colocate_with(op);
                    loop_state.AddWhileContext(op, between_op_list, between_ops);
                }
                pos++;
            }

            return loop_state;
        }

        public static bool IsLoopExit(Operation op)
            => op.OpType == "Exit" || op.OpType == "RefExit";

        public static bool IsLoopSwitch(Operation op)
        {
            if (IsSwitch(op))
            {
                var ctxt = op._get_control_flow_context();
                return ctxt != null && ctxt.IsWhileContext() && !IsCondSwitch(op);
            }
            return false;
        }

        public static bool IsCondSwitch(Operation op)
        {
            throw new NotImplementedException("IsCondSwitch");
        }

        public static bool IsSwitch(Operation op)
            => op.type == "Switch" || op.type == "RefSwitch";

        public static Tensor[] tuple(Tensor[] tensors, string name = null, Operation[] control_inputs = null)
        {
            return tf_with(ops.name_scope(name, "tuple", tensors), scope =>
            {
                name = scope;
                var gating_ops = tensors.Where(x => x != null).Select(x => x.op).ToList();

                if (control_inputs != null)
                {
                    foreach (var c in control_inputs)
                        gating_ops.Add(c);
                }

                // Note that in order to ensure ordering in the pbtxt, we must take care to
                // ensure the order here.
                gating_ops = gating_ops.OrderBy(x => x._id).ToList();
                var gate = group(gating_ops.ToArray());

                var tpl = new List<Tensor>();
                foreach (var t in tensors)
                {
                    if (t != null)
                        tpl.Add(with_dependencies(new Operation[] { gate }, t));
                    else
                        tpl.Add(null);
                }

                return tpl.ToArray();
            });
        }

        internal static Tensor _case_helper(Func<Tensor, Tensor> cond_fn, Tensor[] pred_fn_pairs, Func<Tensor[]> callable_default, bool exclusive, string name,
            bool allow_python_preds = false)
        {
            /*
            (Tensor[] predicates, Tensor[] actions) = _case_verify_and_canonicalize_args(
                pred_fn_pairs, exclusive, name, allow_python_preds);
            return tf_with(ops.name_scope(name, "case", new [] {predicates}), delegate
            {
                if (callable_default == null)
                {
                    (callable_default, predicates, actions) = _case_create_default_action(
                        predicates, actions);
                }
                var fn = callable_default;
            });
            */

            throw new NotImplementedException("_case_helper");
        }

        internal static (Func<Tensor[]>, Tensor[], Tensor[]) _case_create_default_action(Tensor[] predicates, Tensor[] actions)
        {
            throw new NotImplementedException("_case_create_default_action");
        }

        internal static (Tensor[], Tensor[]) _case_verify_and_canonicalize_args(Tensor[] pred_fn_pairs, bool exclusive, string name, bool allow_python_preds)
        {
            throw new NotImplementedException("_case_verify_and_canonicalize_args");
        }

        public static Tensor case_v2(Tensor[] pred_fn_pairs, Func<Tensor[]> callable_default = null, bool exclusive = false, bool strict = false, string name = "case")
            => _case_helper(
                cond_fn: (Tensor x) => cond(x),
                pred_fn_pairs,
                default,
                exclusive,
                name,
                allow_python_preds: false//,
                                         //strict: strict
            );

        /// <summary>
        /// Produces the content of `output_tensor` only after `dependencies`.
        /// 
        /// In some cases, a user may want the output of an operation to be
        /// consumed externally only after some other dependencies have run
        /// first.This function ensures returns `output_tensor`, but only after all
        /// operations in `dependencies` have run.Note that this means that there is
        /// no guarantee that `output_tensor` will be evaluated after any `dependencies`
        /// have run.
        /// 
        /// See also `tf.tuple` and `tf.group`.
        /// </summary>
        /// <param name="dependencies">Iterable of operations to run before this op finishes.</param>
        /// <param name="output_tensor">A `Tensor` or `IndexedSlices` that will be returned.</param>
        /// <param name="name">(Optional) A name for this operation.</param>
        /// <returns>Same as `output_tensor`.</returns>
        public static Tensor with_dependencies(Operation[] dependencies, Tensor output_tensor, string name = null)
        {
            //TODO: missing original code
            //if context.executing_eagerly():
            //    return output_tensor
            return tf_with(ops.name_scope(name, "control_dependency", new { dependencies, output_tensor }), scope =>
            {
                name = scope;
                ops.colocate_with(output_tensor);
                {
                    return tf_with(ops.control_dependencies(dependencies), ctl =>
                    {
                        output_tensor = ops.convert_to_tensor_or_composite(output_tensor);
                        return _Identity(output_tensor, name: name);
                    });
                }
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

        public static void _SetShapeInvariants(Tensor[] input_vars, Tensor[] enter_vars, TensorShape[] shapes = null)
        {
            if (shapes == null)
                return;

            var flat_shapes = nest.flatten2(shapes);
            foreach (var (inp, var, shape) in zip(input_vars, enter_vars, flat_shapes))
            {
                var.set_shape(shape);
            }
        }

        ///  <summary>
        ///  Forwards `data` to an output determined by `pred`.
        ///  If `pred` is false, the `data` input is forwarded to the first output.
        ///  Otherwise, the data goes to the second output.
        ///  
        ///  This op handles `Tensor`s and `IndexedSlices`.
        ///  </summary>
        ///  <param name="data">The tensor to be forwarded to the appropriate output.</param>
        ///  <param name="pred">A scalar that specifies which output port will receive data.</param>
        /// <param name="name"> A name for this operation (optional).</param>
        /// <returns>
        ///  `(output_false, output_true)`: If `pred` is true, data will be forwarded to
        /// `output_true`, otherwise it goes to `output_false`.
        /// </returns>
        public static Tensor[] _SwitchRefOrTensor(Tensor data, Tensor pred, string name = "Switch")
        {
            data = ops.convert_to_tensor_or_composite(data, name: "data");
            // NOTE(vrv): ops.colocate_with(data, ignore_existing=True) below
            // addresses the following scenario.
            //
            // Assume you execute Optimizer.apply_gradients() in a branch of a cond().
            //
            // 1. The update op is created inside a `with ops.colocate(var):` block
            //
            // 2. Some tensor `data` is captured and a switch is created in a
            //    `with ops.colocate_with(data):` block.
            //
            // with ops.colocate_with(var):
            //  with ops.colocate_with(data):
            //    op = ...
            //
            // var and data may be pinned to different devices, so we want to ops
            // created within ops.colocate_with(data) to ignore the existing stack.
            ops.colocate_with(data, ignore_existing: true);
            {
                if (data is Tensor)
                {
                    if (data.dtype.is_ref_dtype())
                        return gen_control_flow_ops.ref_switch(data, pred, name: name);
                }
                return @switch(data, pred, name: name);
            }
        }

        /// <summary>
        /// Return `true_fn()` if the predicate `pred` is true else `false_fn()`.
        /// 
        /// `true_fn` and `false_fn` both return lists of output tensors. `true_fn` and
        /// `false_fn` must have the same non-zero number and type of outputs.
        /// 
        /// **WARNING**: Any Tensors or Operations created outside of `true_fn` and
        /// `false_fn` will be executed regardless of which branch is selected at runtime.
        /// 
        /// Although this behavior is consistent with the dataflow model of TensorFlow,
        /// it has frequently surprised users who expected a lazier semantics.
        /// Consider the following simple program:
        /// 
        /// z = tf.multiply(a, b)
        /// result = tf.cond(x &lt; y, ()=> tf.add(x, z), ()=> tf.square(y))
        /// 
        /// If `x&lt;y`, the `tf.add` operation will be executed and `tf.square`
        /// operation will not be executed.Since `z` is needed for at least one
        /// branch of the `cond`, the `tf.multiply` operation is always executed,
        /// unconditionally.
        /// 
        /// Note that `cond` calls `true_fn` and `false_fn` *exactly once* (inside the
        /// call to `cond`, and not at all during `Session.run()`). `cond`
        /// stitches together the graph fragments created during the `true_fn` and
        /// `false_fn` calls with some additional graph nodes to ensure that the right
        /// branch gets executed depending on the value of `pred`.
        /// 
        /// `tf.cond` supports nested structures as implemented in
        /// `tensorflow.python.util.nest`. Both `true_fn` and `false_fn` must return the
        /// same(possibly nested) value structure of lists, tuples, and/or named tuples.
        /// Singleton lists and tuples form the only exceptions to this: when returned by
        /// `true_fn` and/or `false_fn`, they are implicitly unpacked to single values.
        /// This behavior is disabled by passing `strict= True`.
        /// </summary>
        /// <param name="pred"> A scalar determining whether to return the result of `true_fn` or
        /// `false_fn`.</param>
        /// <param name="true_fn">The callable to be performed if pred is true.</param>
        /// <param name="false_fn">The callable to be performed if pred is false.</param>
        /// <param name="strict"> A boolean that enables/disables 'strict' mode; see above.</param>
        /// <param name="name">Optional name prefix for the returned tensors.</param>
        /// <returns>Tensors returned by the call to either `true_fn` or `false_fn`. If the
        /// callables return a singleton list, the element is extracted from the list.</returns>
        public static Tensor cond(Tensor pred,
            Func<ITensorOrOperation> true_fn = null,
            Func<ITensorOrOperation> false_fn = null,
            string name = null)
        {
            return tf_with(ops.name_scope(name, "cond", new { pred }), delegate
            {
                if (tf.Context.executing_eagerly())
                {
                    if ((bool)pred)
                        return true_fn() as Tensor;
                    else
                        return false_fn() as Tensor;
                }

                // Add the Switch to the graph.
                var switch_result = @switch(pred, pred);
                var (p_2, p_1) = (switch_result[0], switch_result[1]);
                var pivot_1 = array_ops.identity(p_1, name: "switch_t");
                var pivot_2 = array_ops.identity(p_2, name: "switch_f");
                pred = array_ops.identity(pred, name: "pred_id");

                // Disable the fetching of tensors that are only on one branch of cond.
                foreach (var tensor in new Tensor[] { p_1, p_2, pivot_1, pivot_2, pred })
                    tensor.op.graph.prevent_fetching(tensor.op);

                // Build the graph for the true branch in a new context.
                var context_t = new CondContext(pred: pred, pivot: pivot_1, branch: 1);
                ITensorOrOperation orig_res_t;
                Tensor res_t;
                try
                {
                    context_t.Enter();
                    (orig_res_t, res_t) = context_t.BuildCondBranch(true_fn);
                    context_t.ExitResult(new[] { res_t });
                }
                finally
                {
                    context_t.Exit();
                }
                // Build the graph for the false branch in a new context.
                var context_f = new CondContext(pred: pred, pivot: pivot_2, branch: 0);
                ITensorOrOperation orig_res_f;
                Tensor res_f;
                try
                {
                    context_f.Enter();
                    (orig_res_f, res_f) = context_f.BuildCondBranch(false_fn);
                    context_f.ExitResult(new[] { res_f });
                }
                finally
                {
                    context_f.Exit();
                }

                var res_t_flat = new Tensor[] { res_t };
                var res_f_flat = new Tensor[] { res_f };

                var merges = zip(res_f_flat, res_t_flat)
                    .Select(pair => merge(new Tensor[] { pair.Item1, pair.Item2 })[0])
                    .ToArray();

                if (orig_res_t is Tensor orig_res_tensor)
                    merges = _convert_flows_to_tensorarrays(new[] { orig_res_tensor }, merges)
                        .Select(x => x as Tensor)
                        .ToArray();
                else
                {

                }

                if (context_t.outer_context == null)
                {
                    ops.add_to_collection(tf.GraphKeys.COND_CONTEXT, context_t);
                    ops.add_to_collection(tf.GraphKeys.COND_CONTEXT, context_f);
                }

                return merges[0];
            });
        }

        public static Tensor[] cond<T>(Tensor pred,
            Func<T[]> true_fn = null,
            Func<T[]> false_fn = null,
            bool strict = false,
            string name = null)
        {
            return tf_with(ops.name_scope(name, "cond", new { pred }), delegate
            {
                if (tf.Context.executing_eagerly())
                {
                    if (pred.ToArray<bool>()[0])
                        return true_fn() as Tensor[];
                    else
                        return false_fn() as Tensor[];
                }

                // Add the Switch to the graph.
                var switch_result = @switch(pred, pred);
                var p_2 = switch_result[0];
                var p_1 = switch_result[1];
                var pivot_1 = array_ops.identity(p_1, name: "switch_t");
                var pivot_2 = array_ops.identity(p_2, name: "switch_f");
                pred = array_ops.identity(pred, name: "pred_id");

                // Disable the fetching of tensors that are only on one branch of cond.
                foreach (var tensor in new Tensor[] { p_1, p_2, pivot_1, pivot_2, pred })
                    tensor.op.graph.prevent_fetching(tensor.op);

                // Build the graph for the true branch in a new context.
                var context_t = new CondContext(pred: pred, pivot: pivot_1, branch: 1);
                context_t.Enter();
                var (orig_res_t, res_t) = context_t.BuildCondBranch(true_fn);
                context_t.ExitResult(res_t);
                context_t.Exit();

                // Build the graph for the false branch in a new context.
                var context_f = new CondContext(pred: pred, pivot: pivot_2, branch: 0);
                context_f.Enter();
                var (orig_res_f, res_f) = context_f.BuildCondBranch(false_fn);
                context_f.ExitResult(res_f);
                context_f.Exit();

                var res_t_flat = res_t;
                var res_f_flat = res_f;

                var merges = zip(res_f_flat, res_t_flat)
                    .Select(pair => merge(new[] { pair.Item1, pair.Item2 })[0])
                    .ToArray();

                if (orig_res_t is Tensor[] orig_res_tensor)
                    merges = _convert_flows_to_tensorarrays(orig_res_tensor, merges)
                        .Select(x => x as Tensor)
                        .ToArray();
                else if (orig_res_t is float[] orig_res_float)
                {

                }
                else
                {

                }

                if (context_t.outer_context == null)
                {
                    ops.add_to_collection(tf.GraphKeys.COND_CONTEXT, context_t);
                    ops.add_to_collection(tf.GraphKeys.COND_CONTEXT, context_f);
                }

                return merges;
            });
        }

        public static ITensorOrTensorArray[] _convert_flows_to_tensorarrays(ITensorOrTensorArray[] tensors_or_tensorarrays, Tensor[] tensors_or_flows)
        {
            return zip(tensors_or_tensorarrays, tensors_or_flows).Select(x =>
            {
                var (ta, t_or_flow) = (x.Item1, x.Item2);
                if (ta is TensorArray ta_1)
                    return tensor_array_ops.build_ta_with_new_flow(ta_1, t_or_flow) as ITensorOrTensorArray;
                else
                    return t_or_flow as ITensorOrTensorArray;
            }).ToArray();
        }

        /// <summary>
        /// Returns the value of an available element of `inputs`.
        /// 
        /// This op tests each of the tensors in `inputs` in turn to determine if any of
        /// them is available.If it finds an available tensor, it returns it and its
        /// index in `inputs`.
        /// 
        /// It is an error if more than one tensor in `inputs` is available.If no tensor
        /// in `inputs` is available, the returned tensor and index are not set.
        /// 
        /// This op handles both `Tensor`s and `IndexedSlices`. If inputs has a mix of
        /// `Tensor`s and `IndexedSlices`, all inputs are converted to IndexedSlices
        /// before merging.
        /// </summary>
        /// <param name="inputs">inputs: The input tensors, at most one of which is available.</param>
        /// <param name="name">A name for this operation (optional).</param>
        /// <returns></returns>
        public static MergeOutput merge(Tensor[] inputs, string name = null)
        {
            if (inputs.Any(x => x == null))
                throw new ValueError($"At least one of the merge inputs is null: {inputs}");
            return tf_with(ops.name_scope(name, "Merge", inputs), scope =>
            {
                name = scope;
                inputs = inputs.Select(inp =>
                            ops.internal_convert_to_tensor_or_indexed_slices(inp, as_ref: true))
                        .ToArray();
                return gen_control_flow_ops.merge(inputs, name);
            });
        }

        /// <summary>
        /// Forwards `data` to an output determined by `pred`.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="pred"></param>
        /// <param name="dtype"></param>
        /// <param name="name"></param>
        public static Tensor[] @switch(Tensor data,
            Tensor pred,
            TF_DataType dtype = TF_DataType.DtInvalid,
            string name = null)
        {
            return tf_with(ops.name_scope(name, "Switch", new { data, pred }), scope =>
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
            else
            {
                var op_ctxt = op._get_control_flow_context();
                if (op_ctxt != null)
                {
                    // We are in a cond context. Use a switch to create zeros only when needed.
                    var pred = op_ctxt.pred;
                    var branch = op_ctxt.branch;
                    var switch_val = @switch(op.inputs[0], pred)[1 - branch];
                    var pivot = array_ops.identity(switch_val);
                    if (val.dtype == dtypes.resource)
                        throw new NotImplementedException("");
                    var zeros_shape = array_ops.shape_internal(switch_val, optimize: false);
                    // Ensure ops created within array_ops.zeros are dominated by switch in
                    // cond context.
                    return tf_with(ops.control_dependencies(new[] { pivot }), delegate
                    {
                        return array_ops.zeros(zeros_shape, dtype: val.dtype);
                    });
                }
                else
                {
                    return array_ops.zeros_like(val, optimize: false);
                }
            }
        }

        public static Tensor[] while_loop(Func<Tensor[], Tensor> cond,
            Func<Tensor[], Tensor[]> body,
            Tensor[] loop_vars,
            int parallel_iterations = 10,
            string name = null)
        {
            var executing_eagerly = tf.Context.executing_eagerly();
            if (!executing_eagerly)
            {
                throw new NotImplementedException("");
            }

            return tf_with(ops.name_scope("name", "while"), delegate
            {
                while ((bool)cond(loop_vars))
                {
                    loop_vars = body(loop_vars);
                }

                return loop_vars;
            });
        }

        /// <summary>
        /// Repeat `body` while the condition `cond` is true.
        /// </summary>
        /// <param name="cond"></param>
        /// <param name="body"></param>
        /// <param name="loop_vars"></param>
        /// <param name="shape_invariants"></param>
        public static TItem while_loop<TItem>(Func<TItem, Tensor> cond, Func<TItem, TItem> body, TItem loop_vars,
            TensorShape[] shape_invariants = null,
            int parallel_iterations = 10,
            bool back_prop = true,
            bool swap_memory = false,
            string name = null,
            Tensor maximum_iterations = null,
            bool return_same_structure = false) where TItem : IFromMergeVars<TItem>, new()
        {
            return tf_with(ops.name_scope(name, "while", loop_vars), scope =>
            {
                if (loop_vars == null)
                    throw new ValueError("No loop variables provided");
                if (cond == null)
                    throw new ValueError("cond must be callable.");
                if (body == null)
                    throw new ValueError("body must be callable.");
                if (parallel_iterations < 1)
                    throw new ValueError("parallel_iterations must be a positive integer.");

                var try_to_pack = loop_vars is Tensor && !return_same_structure;
                var counter = constant_op.constant(0, dtype: maximum_iterations.dtype, name: "iteration_counter");
                var orig_cond = cond;
                var orig_body = body;

                LoopVar<TItem> loop_vars_1 = null;
                Func<LoopVar<TItem>, LoopVar<TItem>> body_buildloop = null;
                Func<LoopVar<TItem>, Tensor> cond_buildloop = null;

                if (try_to_pack)
                {

                }
                else
                {
                    loop_vars_1 = new LoopVar<TItem>(counter, loop_vars);
                    cond_buildloop = (item) =>
                    {
                        var (i, lv) = (item.Counter, item.Item);
                        var oc = orig_cond(lv);
                        return math_ops.logical_and(i < maximum_iterations, oc);
                    };

                    body_buildloop = (item) =>
                    {
                        var (i, lv) = (item.Counter, item.Item);
                        var ob = orig_body(lv);
                        return new LoopVar<TItem>(i + 1, ob);
                    };
                }
                try_to_pack = false;

                var loop_context = new WhileContext(
                    maximum_iterations: maximum_iterations,
                    parallel_iterations: parallel_iterations,
                    back_prop: back_prop,
                    swap_memory: swap_memory);

                if (loop_context.outer_context == null)
                    ops.add_to_collection(tf.GraphKeys.WHILE_CONTEXT, loop_context);

                var results = loop_context.BuildLoop(cond_buildloop, body_buildloop, loop_vars_1, shape_invariants,
                                    return_same_structure);

                //if (maximum_iterations != null)
                return results.Item;
                //else
                //return results;
            });
        }

        /// <summary>
        /// Creates or finds a child frame, and makes `data` available to it.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="frame_name"></param>
        /// <param name="is_constant"></param>
        /// <param name="parallel_iterations"></param>
        /// <param name="use_ref"></param>
        /// <param name="use_input_shape"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor _Enter(Tensor data, string frame_name,
            bool is_constant = false,
            int parallel_iterations = 10,
            bool use_ref = true,
            bool use_input_shape = true,
            string name = null)
        {
            Tensor result;
            data = ops.internal_convert_to_tensor_or_indexed_slices(data, as_ref: true);
            if (data.dtype.is_ref_dtype() && use_ref)
                throw new NotImplementedException("_Enter");
            else
                result = gen_control_flow_ops.enter(
                    data, frame_name, is_constant, parallel_iterations, name: name);

            if (use_input_shape)
                result.set_shape(data.TensorShape);

            return result;
        }
    }
}
