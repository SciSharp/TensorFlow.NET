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

using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Tensorflow.Functions;
using Tensorflow.Gradients;
using Tensorflow.Graphs;
using Tensorflow.Operations;
using Tensorflow.Operations.ControlFlows;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class gradients_util
    {
        // Represents the output of TFE_Py_TapeSetPossibleGradientTypes. Real enums are
        // unfortunately too slow to use here.
        public static int POSSIBLE_GRADIENT_TYPES_NONE = 0;
        public static int POSSIBLE_GRADIENT_TYPES_FIRST_ORDER = 1;
        public static int POSSIBLE_GRADIENT_TYPES_HIGHER_ORDER = 2;
        public static Tensor[] _GradientsHelper(Tensor[] ys,
            Tensor[] xs,
            Tensor[] grad_ys = null,
            string name = "gradients",
            bool colocate_gradients_with_ops = false,
            bool gate_gradients = false,
            int aggregation_method = 0,
            Tensor[] stop_gradients = null,
            Graph src_graph = null)
        {
            if (src_graph == null)
                src_graph = ops.get_default_graph();

            // If src_graph is a _FuncGraph (i.e. a function body), gather it and all
            // ancestor graphs. This is necessary for correctly handling captured values.
            var func_graphs = new List<FuncGraph>();
            var curr_graph = src_graph;
            if (src_graph is FuncGraph func_graph)
            {
                func_graphs.append(func_graph);
                curr_graph = func_graph.OuterGraph;
            }
                

            if (stop_gradients == null)
                stop_gradients = new Tensor[0];
            if (grad_ys == null)
                grad_ys = new Tensor[ys.Length];

            // Iterate over the collected ops.
            /*
             * grads: op => list of gradients received on each output endpoint of the
             * op.  The gradients for each endpoint are initially collected as a list.
             * When it is time to call the op's gradient function, for each endpoint we
             * aggregate the list of received gradients into a Add() Operation if there
             * is more than one.
             */
            var grads = new Dictionary<string, List<List<Tensor>>>();
            Operation[] reachable_to_ops = null;
            ControlFlowState loop_state = null;
            Dictionary<string, int> pending_count = null;

            tf_with(ops.name_scope(name, "gradients",
                values: ys.Concat(xs).Concat(stop_gradients).Concat(grad_ys)), scope =>
                {
                    string grad_scope = scope;
                    // Get a uid for this call to gradients that can be used to help
                    // cluster ops for compilation.
                    var gradient_uid = curr_graph.unique_name("uid");
                    ys = ops.convert_n_to_tensor_or_indexed_slices(ys, name: "y");
                    xs = ops.internal_convert_n_to_tensor_or_indexed_slices(xs, name: "x", as_ref: true);
                    grad_ys = _DefaultGradYs(grad_ys, ys, colocate_gradients_with_ops, gradient_uid);

                    /*
                     * The approach we take here is as follows: Create a list of all ops in the
                     * subgraph between the ys and xs.  Visit these ops in reverse order of ids
                     * to ensure that when we visit an op the gradients w.r.t its outputs have
                     * been collected.  Then aggregate these gradients if needed, call the op's
                     * gradient function, and add the generated gradients to the gradients for
                     * its input.
                     */

                    // Initialize the pending count for ops in the connected subgraph from ys
                    // to the xs.
                    var to_ops = ys.Select(x => x.op).ToList();
                    var from_ops = xs.Select(x => x.op).ToList();
                    var stop_gradient_ops = stop_gradients.Select(x => x.op).ToList();
                    (reachable_to_ops, pending_count, loop_state) = _PendingCount(to_ops, from_ops, colocate_gradients_with_ops, func_graphs , xs);

                    // Add the initial gradients for the ys.
                    foreach (var (y, grad_y) in zip(ys, grad_ys))
                        _SetGrad(grads, y, grad_y);

                    // Initialize queue with to_ops.
                    var queue = new Queue<Operation>();
                    // Add the ops in 'to_ops' into the queue.
                    var to_ops_set = new List<Operation>();
                    foreach (var op in to_ops)
                    {
                        // 'ready' handles the case where one output gradient relies on
                        // another output's gradient.
                        if (!pending_count.ContainsKey(op.name))
                            pending_count[op.name] = 0;
                        bool ready = pending_count[op.name] == 0;
                        if (ready && !to_ops_set.Contains(op) && reachable_to_ops.Contains(op))
                        {
                            to_ops_set.Add(op);
                            queue.Enqueue(op);
                        }
                    }

                    if (loop_state != null)
                    {
                        var loop_exits = loop_state.ProcessUnusedLoopExits(pending_count, to_ops_set);
                        foreach (var y in loop_exits)
                        {
                            //if(IsTrainable(y))
                            throw new NotImplementedException("");
                        }
                    }

                    var stop_ops = _StopOps(from_ops, stop_gradient_ops, pending_count, xs);
                    while (queue.Count > 0)
                    {
                        // generate gradient subgraph for op.
                        var op = queue.Dequeue();

                        _maybe_colocate_with(op, gradient_uid, colocate_gradients_with_ops);
                        {
                            if (loop_state != null)
                                loop_state.EnterGradWhileContext(op, before: true);
                            var out_grads = _AggregatedGrads(grads, op, gradient_uid, loop_state, aggregation_method);
                            if (loop_state != null)
                                loop_state.ExitGradWhileContext(op, before: true);

                            Tensor[] in_grads = null;
                            Func<Operation, Tensor[], Tensor[]> grad_fn = null;
                            var is_partitioned_call = _IsPartitionedCall(op);
                            var is_func_call = src_graph.IsFunction(op.type) || is_partitioned_call;
                            var has_out_grads = out_grads.Exists(x => x != null);
                            if (has_out_grads && !stop_ops.Contains(op))
                            {
                                // A grad_fn must be defined, either as a function or as None
                                // for ops that do not have gradients.
                                try
                                {
                                    grad_fn = ops.get_gradient_function(op);
                                }
                                catch (LookupError)
                                {
                                    if (is_func_call)
                                    {
                                        EagerDefinedFunction func_call = null;
                                        if (is_partitioned_call)
                                        {
                                            var func_attr = op.get_attr("f");
                                            Debug.Assert(func_attr is NameAttrList);
                                            var func_name = ((NameAttrList)func_attr).Name;
                                            func_call = src_graph._get_function(func_name);
                                            if(func_call is null && src_graph.OuterGraph is not null)
                                            {
                                                var graph = src_graph.OuterGraph;
                                                while(graph is not null)
                                                {
                                                    func_call = graph._get_function(func_name);
                                                    if(func_call is not null)
                                                    {
                                                        break;
                                                    }
                                                    if(graph.OuterGraph is not null)
                                                    {
                                                        graph = graph.OuterGraph;
                                                    }
                                                    else
                                                    {
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                        else
                                        {
                                            func_call = src_graph._get_function(op.type);
                                        }
                                        // skip the following codes:
                                        // `func_call = getattr(op, "__defun", func_call)`
                                        grad_fn = func_call.csharp_grad_func;
                                    }
                                    else
                                    {
                                        throw new LookupError($"No gradient defined for operation '{op.name}' (op type: {op.type})");
                                    }
                                }
                            }

                            if (loop_state != null)
                                loop_state.EnterGradWhileContext(op, before: false);

                            if ((is_func_call || grad_fn != null) && has_out_grads)
                            {
                                // NOTE: If _AggregatedGrads didn't compute a value for the i'th
                                // output, it means that the cost does not depend on output[i],
                                // therefore dC/doutput[i] is 0.
                                foreach (var (i, out_grad) in enumerate(out_grads))
                                {
                                    if (out_grad == null &&
                                        (grad_fn == null || _IsTrainable(op.outputs[i])))
                                    {
                                        // Only trainable outputs or outputs for a function call that
                                        // will use SymbolicGradient get a zero gradient. Gradient
                                        // functions should ignore the gradient for other outputs.
                                        if (loop_state != null)
                                            out_grads[i] = new List<Tensor> { loop_state.ZerosLike(op, i) };
                                        else
                                            out_grads[i] = new List<Tensor> { control_flow_ops.ZerosLikeOutsideLoop(op, i) };
                                    }
                                }

                                tf_with(ops.name_scope(op.name + "_grad"), scope1 =>
                                {
                                    if (grad_fn != null)
                                    {
                                        in_grads = _MaybeCompile(grad_scope,
                                            op,
                                            out_grads.Where(x => x != null).Select(x => x[0]).ToArray(),
                                            null,
                                            grad_fn);
                                    }
                                    else
                                    {
                                        in_grads = _MaybeCompile(grad_scope, op, out_grads.Where(x => x != null).Select(x => x[0]).ToArray(),
                                            null, (x, y) => _SymGrad(x, y));
                                        throw new NotImplementedException("lambda: _SymGrad(op, out_grads)");
                                    }
                                    _VerifyGeneratedGradients(in_grads, op);
                                    if (gate_gradients && in_grads.Count(x => x != null) > 1)
                                    {
                                        ops._colocate_with_for_gradient(null, gradient_uid, ignore_existing: true);
                                        in_grads = control_flow_ops.tuple(in_grads);
                                    }
                                });
                            }
                            else
                            {
                                // If no grad_fn is defined or none of out_grads is available,
                                // just propagate a list of None backwards.
                                in_grads = new Tensor[_NonEagerInputs(op, xs).Count()];
                            }

                            var inputs = _NonEagerInputs(op, xs).ToList();
                            foreach (var (t_in, in_grad) in zip(inputs, in_grads))
                            {
                                if (in_grad != null)
                                {
                                    if (!(in_grad is null) &&
                                        in_grad.Tag == null && // maybe a IndexedSlice
                                        t_in.dtype != TF_DataType.TF_RESOURCE)
                                    {
                                        in_grad.shape = t_in.shape;
                                    }

                                    _SetGrad(grads, t_in, in_grad);
                                }
                            }

                            if (loop_state != null)
                                loop_state.ExitGradWhileContext(op, before: false);
                        }

                        // Update pending count for the inputs of op and enqueue ready ops.
                        _UpdatePendingAndEnqueueReady(grads, op, queue, pending_count, loop_state, xs);
                    }
                });

            if (loop_state != null)
                loop_state.PostProcessing();
            return xs.Select(x => _GetGrad(grads, x)).ToArray();
        }

        /// <summary>
        /// Fill in default values for grad_ys.
        /// </summary>
        /// <param name="grad_ys">List of gradients, can contain None.</param>
        /// <param name="ys">List of tensors.</param>
        /// <param name="colocate_gradients_with_ops"></param>
        /// <param name="gradient_uid"></param>
        private static Tensor[] _DefaultGradYs(Tensor[] grad_ys, Tensor[] ys, bool colocate_gradients_with_ops, string gradient_uid = "__unsupported__")
        {
            var new_grad_ys = new List<Tensor>();

            foreach(var (i, (y, grad_y)) in enumerate(zip(ys, grad_ys)))
            {
                _maybe_colocate_with(y.op, gradient_uid, colocate_gradients_with_ops);

                if (grad_y == null)
                {
                    if (y.dtype.is_complex())
                        throw new TypeAccessException($"Gradients of complex tensors must set grad_ys (y.dtype = {y.dtype})");
                    var shape = array_ops.shape(y);
                    var constant = constant_op.constant(1, y.dtype, name: $"grad_ys_{i}");
                    var fill = gen_array_ops.fill(shape, constant);
                    new_grad_ys.append(fill);
                    continue;
                }

                if (y.dtype.is_floating() || y.dtype.is_integer())
                {

                }

                // Create a grad_y tensor in the name scope of the gradient.
                new_grad_ys.append(array_ops.identity(grad_y, name: $"grad_ys_{i}"));
            }

            return new_grad_ys.ToArray();
        }

        private static void _maybe_colocate_with(Operation op, string gradient_uid, bool colocate_gradients_with_ops)
        {

        }

        /// <summary>
        /// Initialize the pending count for ops between two lists of Operations.
        /// 'pending_count[op]' indicates the number of backprop inputs
        /// to this operation.
        /// </summary>
        /// <param name="to_ops"></param>
        /// <param name="from_ops"></param>
        /// <param name="colocate_gradients_with_ops"></param>
        /// <param name="func_graphs"></param>
        /// <param name="xs"></param>
        private static (Operation[], Dictionary<string, int>, ControlFlowState) _PendingCount(List<Operation> to_ops, 
            List<Operation> from_ops, 
            bool colocate_gradients_with_ops, 
            List<FuncGraph> func_graphs, 
            Tensor[] xs)
        {
            // Mark reachable ops from from_ops.
            var reached_ops = new List<Operation>();
            _MarkReachedOps(from_ops, reached_ops, func_graphs);
            // X in reached_ops iff X is reachable from from_ops by a path of zero or more
            // backpropagatable tensors.

            var reachable_to_ops = to_ops.Where(x => reached_ops.Contains(x)).Select(x => x).ToArray();

            var between_ops = new List<Operation>();
            var between_op_list = new List<Operation>();

            Queue<Operation> queue = new Queue<Operation>(to_ops);
            while (queue.Count > 0)
            {
                var op = queue.Dequeue();

                if (reached_ops.Contains(op))
                {
                    between_ops.Add(op);
                    between_op_list.Insert(between_op_list.Count, op);
                    // Clear the boolean so we won't add the inputs again.
                    reached_ops.Remove(op);
                    foreach (var inp in _NonEagerInputs(op, xs))
                        queue.Enqueue(inp.op);
                }
            }
            // X in between_ops iff X is on a path of zero or more backpropagatable tensors
            // between from_ops and to_ops

            // 'loop_state' is None if there are no while loops.
            var loop_state = control_flow_ops.MaybeCreateControlFlowState(between_op_list, between_ops, colocate_gradients_with_ops);

            // Initialize pending count for between ops.
            var pending_count = new Dictionary<string, int>();
            foreach (var op in between_op_list)
            {
                foreach (Tensor x in _NonEagerInputs(op, xs))
                {
                    if (between_ops.Contains(x.op))
                    {
                        if (!pending_count.ContainsKey(x.op.name))
                            pending_count[x.op.name] = 0;

                        pending_count[x.op.name] += 1;
                    }
                }
            }

            return (reachable_to_ops.ToArray(), pending_count, loop_state);
        }

        /// <summary>
        /// Sets gradient "grad" in "grads" for tensor "t".
        /// </summary>
        /// <param name="grads"></param>
        /// <param name="t"></param>
        /// <param name="grad"></param>
        private static void _SetGrad(Dictionary<string, List<List<Tensor>>> grads, Tensor t, Tensor grad)
        {
            var op = t.op;
            var op_grads = grads.ContainsKey(op.name) ? grads[op.name] : null;
            if (op_grads == null)
            {
                op_grads = op.outputs.Select(x => new List<Tensor>()).ToList();
                grads[op.name] = op_grads;
            }
            var t_grads = op_grads[t.value_index];
            if (t_grads.Count > 0 &&
                control_flow_util.IsLoopSwitch(op))
                op_grads[t.value_index][0] = grad;
            else
                t_grads.Add(grad);
        }

        private static IEnumerable<Tensor> _NonEagerInputs(Operation op, Tensor[] xs)
        {
            for (int i = 0; i < op.inputs.Length; i++)
                yield return op.inputs[i];
        }

        private static List<List<Tensor>> _AggregatedGrads(Dictionary<string, List<List<Tensor>>> grads, Operation op, string gradient_uid,
            ControlFlowState loop_state, int aggregation_method = 0)
        {
            var out_grads = _GetGrads(grads, op);

            foreach (var (i, out_grad) in enumerate(out_grads))
            {
                if (loop_state != null)
                {
                    if (out_grads.Count > 1 &&
                        out_grads[1].Count > 0 &&
                        control_flow_util.IsLoopSwitch(op))
                        continue;
                }

                // Aggregate multiple gradients, and convert [] to None.
                if (out_grad.Count > 0)
                {
#pragma warning disable CS0219 // Variable is assigned but its value is never used
                    string used = "";
#pragma warning restore CS0219 // Variable is assigned but its value is never used
                    if (out_grad.Count < 2)
                    {
                        used = "nop";
                        if (out_grad.Count == 0)
                        {
                            throw new ValueError("_AggregatedGrads out_grad.Length == 0");
                        }

                        out_grads[i] = new List<Tensor> { out_grad[0] };
                    }
                    else
                    {
                        used = "add_n";
                        out_grads[i] = new List<Tensor> { _MultiDeviceAddN(out_grad.ToArray(), gradient_uid) };
                    }
                }
                else
                {
                    out_grads[i] = null;
                }
            }

            return out_grads;
        }

        /// <summary>
        /// Adds tensors from potentially multiple devices.
        /// </summary>
        /// <param name="tensor_list"></param>
        /// <param name="gradient_uid"></param>
        /// <returns></returns>
        private static Tensor _MultiDeviceAddN(Tensor[] tensor_list, string gradient_uid)
        {
            // Basic function structure comes from control_flow_ops.group().
            // Sort tensors according to their devices.
            var tensors_on_device = new Dictionary<string, List<Tensor>>();

            foreach (var tensor in tensor_list)
            {
                if (!tensors_on_device.ContainsKey(tensor.Device))
                    tensors_on_device[tensor.Device] = new List<Tensor>();

                tensors_on_device[tensor.Device].Add(tensor);
            }

            // For each device, add the tensors on that device first.
            var summands = new List<Tensor>();
            foreach (var dev in tensors_on_device.Keys)
            {
                var tensors = tensors_on_device[dev];
                ops._colocate_with_for_gradient(tensors[0].op, gradient_uid, ignore_existing: true);
                summands.Add(math_ops.add_n(tensors.ToArray()));
            }

            return math_ops.add_n(summands.ToArray());
        }

        /// <summary>
        /// The set of ops that terminate the gradient computation.
        /// </summary>
        /// <param name="from_ops">list of Operations.</param>
        /// <param name="stop_gradient_ops">list of Operations never to backprop through.</param>
        /// <param name="pending_count">mapping from operation to number of backprop inputs.</param>
        /// <param name="xs">list of Tensors.</param>
        /// <returns>The set of operations.</returns>
        private static Operation[] _StopOps(List<Operation> from_ops, List<Operation> stop_gradient_ops, Dictionary<string, int> pending_count, Tensor[] xs)
        {
            var stop_ops = new List<Operation>();

            foreach (var op in from_ops)
            {
                bool is_stop_op = true;
                foreach (var inp in _NonEagerInputs(op, xs))
                {
                    if (!pending_count.ContainsKey(inp.op.name))
                        pending_count[inp.op.name] = 0;

                    if (pending_count[inp.op.name] > 0)
                    {
                        is_stop_op = false;
                        break;
                    }
                }
                if (is_stop_op)
                    stop_ops.Insert(0, op);
            }
            stop_ops.AddRange(stop_gradient_ops.Where(x => !stop_ops.Contains(x)));
            return stop_ops.ToArray();
        }

        private static Tensor _GetGrad(Dictionary<string, List<List<Tensor>>> grads, Tensor t)
        {
            var op = t.op;
            if (!grads.ContainsKey(op.name))
                return null;
            var op_grads = grads[op.name];
            var t_grad = op_grads[t.value_index];
            return t_grad[0];
        }

        private static List<List<Tensor>> _GetGrads(Dictionary<string, List<List<Tensor>>> grads, Operation op)
        {
            if (grads.ContainsKey(op.name))
                return grads[op.name];
            else
                return op.outputs.Select(x => new List<Tensor>()).ToList();
        }

        /// <summary>
        /// Mark all ops reached from "from_ops"
        /// </summary>
        /// <param name="from_ops"></param>
        /// <param name="reached_ops"></param>
        /// <param name="func_graphs"></param>
        private static void _MarkReachedOps(List<Operation> from_ops, List<Operation> reached_ops, List<FuncGraph> func_graphs)
        {
            Queue<Operation> queue = new Queue<Operation>(from_ops);
            while (queue.Count > 0)
            {
                var op = queue.Dequeue();

                if (!reached_ops.Contains(op))
                {
                    reached_ops.Add(op);
                    foreach (var output in op.outputs)
                    {
                        if (_IsBackpropagatable(output))
                        {
                            var c = output.consumers().ToList();
                            c.ForEach(x => queue.Enqueue(x));
                        }
                    }
                }
            }
        }

        private static bool _IsBackpropagatable(Tensor tensor)
        {
            if (_IsTrainable(tensor))
            {
                return true;
            }
            else
            {
                var dtype = tensor.dtype.as_base_dtype();
                return new TF_DataType[] { TF_DataType.TF_BFLOAT16, TF_DataType.TF_VARIANT }.Contains(dtype);
            }
        }

        private static bool _IsTrainable(Tensor tensor)
        {
            var dtype = tensor.dtype.as_base_dtype();
            return new TF_DataType[] {TF_DataType.TF_HALF, TF_DataType.TF_FLOAT, TF_DataType.TF_DOUBLE,
                TF_DataType.TF_COMPLEX64, TF_DataType.TF_COMPLEX128, TF_DataType.TF_RESOURCE}.Contains(dtype);
        }

        private static bool _IsPartitionedCall(Operation op)
        {
            return op.OpType == "PartitionedCall" || op.OpType == "StatefulPartitionedCall";
        }

        /// <summary>
        /// Update pending count for the inputs of op and enqueue ready ops.
        /// </summary>
        /// <param name="grads"></param>
        /// <param name="op"></param>
        /// <param name="queue"></param>
        /// <param name="pending_count"></param>
        /// <param name="loop_state"></param>
        /// <param name="xs"></param>
        private static void _UpdatePendingAndEnqueueReady(Dictionary<string, List<List<Tensor>>> grads,
            Operation op,
            Queue<Operation> queue,
            Dictionary<string, int> pending_count,
            ControlFlowState loop_state,
            Tensor[] xs)
        {
            foreach (var x in _NonEagerInputs(op, xs))
            {
                if (!pending_count.ContainsKey(x.op.name))
                    pending_count[x.op.name] = 0;

                pending_count[x.op.name] -= 1;

                var ready = pending_count[x.op.name] == 0;

                if (loop_state != null && !ready)
                {
                    ready = pending_count[x.op.name] > 0 && control_flow_util.IsLoopSwitch(x.op);
                }

                if (ready)
                {
                    // if x is an exit without real gradient, defer processing them.
                    if (control_flow_util.IsLoopExit(x.op))
                    {
                        var grad_state = loop_state.GetGradState(x.op, before: false);
                        grad_state.deferred_exits.append(x);
                        grad_state.pending_exits_count -= 1;
                        // We now have all the exits so process them.
                        if (grad_state.pending_exits_count == 0)
                        {
                            var has_not_none_grad = false;
                            foreach (var y in grad_state.deferred_exits)
                            {
                                if (_HasAnyNotNoneGrads(grads, y.op))
                                {
                                    has_not_none_grad = true;
                                    queue.Enqueue(y.op);
                                }
                                else
                                    grad_state.unused_exits.append(y);
                            }
                            if (has_not_none_grad)
                            {
                                // For an unused exit, if it has trainable outputs, backprop
                                // a zero gradient. Otherwise, just ignore it.
                                foreach (var y in grad_state.unused_exits)
                                {
                                    if (IsTrainable(y))
                                        _SetGrad(grads, y, loop_state.ZerosLikeForExit(y));
                                    queue.Enqueue(y.op);
                                }
                            }
                            else
                            {
                                // All exits are "unused" so use None as gradient.
                                foreach (var y in grad_state.unused_exits)
                                    queue.Enqueue(y.op);
                            }
                        }
                    }
                    else
                    {
                        queue.Enqueue(x.op);
                    }
                }
            }
        }

        public static bool IsTrainable(Tensor tensor)
        {
            var dtype = tensor.dtype.as_base_dtype();
            return new TF_DataType[] { dtypes.float16, dtypes.float32, dtypes.float64,
                              dtypes.complex64, dtypes.complex128,
                              dtypes.resource, dtypes.variant}.Contains(dtype);
        }

        public static int PossibleTapeGradientTypes(Tensor[] tensors)
        {
            return tf.Runner.TFE_TapeSetPossibleGradientTypes(tensors);
        }

        /// <summary>
        /// Return true if op has real gradient.
        /// </summary>
        /// <param name="grads"></param>
        /// <param name="op"></param>
        /// <returns></returns>
        private static bool _HasAnyNotNoneGrads(Dictionary<string, List<List<Tensor>>> grads, Operation op)
        {
            var out_grads = _GetGrads(grads, op);
            foreach (var out_grad in out_grads)
            {
                if (out_grad.Exists(g => g != null))
                    return true;
            }
            return false;
        }


        private static Tensor[] _MaybeCompile(string scope, Operation op, Tensor[] out_grads, Action func, Func<Operation, Tensor[], Tensor[]> grad_fn)
        {
            // scope = scope.TrimEnd('/').Replace('/', '_');
            return grad_fn(op, out_grads);
        }

        private static void _VerifyGeneratedGradients(Tensor[] grads, Operation op)
        {
            if (op.type == "While" || op.type == "StatelessWhile")
                return;

            if (grads.Count() != op.inputs._inputs.Count())
                throw new ValueError($"Num gradients {grads.Length} generated for op {op.node_def} do not match num " +
                    $"inputs {op.inputs._inputs.Count()}");
        }

        private static Tensor[] _SymGrad(Operation op, Tensor[] out_grads)
        {
            var f_in = ((Tensor[])op.inputs).Concat(out_grads).ToArray();
            var f_types = ((Tensor[])op.inputs).Select(x => default_gradient.get_zeros_dtype(x)).ToArray();
            NameAttrList f = new();
            if (_IsPartitionedCall(op))
            {
                var func_attr = op.get_attr("f");
                Debug.Assert(func_attr is NameAttrList);
                f.Name = ((NameAttrList)func_attr).Name;
            }
            else
            {
                f.Name = op.type;
            }
            foreach(var k in op.node_def.Attr.Keys)
            {
                f.Attr[k] = AttrValue.Parser.ParseFrom(op.node_def.Attr[k].ToByteArray());
            }
            var in_grads = gen_functional_ops.symbolic_gradient(f_in, f_types, f);
            return in_grads;
        }
    }
}
