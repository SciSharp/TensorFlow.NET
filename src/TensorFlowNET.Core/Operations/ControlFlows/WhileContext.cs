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
using Tensorflow.Operations.ControlFlows;
using Tensorflow.Util;
using static Tensorflow.control_flow_ops;
using static Tensorflow.Binding;

namespace Tensorflow.Operations
{
    /// <summary>
    /// Creates a `WhileContext`.
    /// </summary>
    public class WhileContext : ControlFlowContext
    {
        bool _back_prop=true;
        GradLoopState _grad_state =null;
        Tensor _maximum_iterations;
        int _parallel_iterations;
        bool _swap_memory;
        Tensor _pivot_for_pred;
        Tensor _pivot_for_body;
        List<Tensor> _loop_exits;
        List<Tensor> _loop_enters;
        Graph _graph;
        public override GradLoopState grad_state => _grad_state;
        public override bool back_prop => _back_prop;

        public WhileContext(Tensor maximum_iterations = null,
            int parallel_iterations = 10,
            bool back_prop = true,
            bool swap_memory = false,
            string name = "while_context",
            GradLoopState grad_state = null,
            WhileContextDef context_def = null,
            string import_scope = null)
        {
            if (context_def != null)
            {
                _init_from_proto(context_def, import_scope: import_scope);
            }
            else
            {
                __init__();
                _init_from_args(maximum_iterations, parallel_iterations, back_prop, swap_memory, name);
            }

            _grad_state = grad_state;
        }

        private void _init_from_args(Tensor maximum_iterations,
            int parallel_iterations,
            bool back_prop,
            bool swap_memory,
            string name)
        {
            _name = ops.get_default_graph().unique_name(name);
            _maximum_iterations = maximum_iterations;
            _parallel_iterations = parallel_iterations;
            _back_prop = back_prop;
            _swap_memory = swap_memory;
            _loop_exits = new List<Tensor>();
            _loop_enters = new List<Tensor>();
            _graph = ops.get_default_graph();
        }

        private void _init_from_proto(WhileContextDef context_def, string import_scope = null)
        {
            var g = ops.get_default_graph();
            _name = ops.prepend_name_scope(context_def.ContextName, import_scope);
            if (!string.IsNullOrEmpty(context_def.MaximumIterationsName))
                _maximum_iterations = g.as_graph_element(ops.prepend_name_scope(context_def.MaximumIterationsName, import_scope)) as Tensor;
            _parallel_iterations = context_def.ParallelIterations;
            _back_prop = context_def.BackProp;
            _swap_memory = context_def.SwapMemory;
            _pivot_for_pred = g.as_graph_element(ops.prepend_name_scope(context_def.PivotForPredName, import_scope)) as Tensor;
            // We use this node to control constants created by the body lambda.
            _pivot_for_body = g.as_graph_element(ops.prepend_name_scope(context_def.PivotForBodyName, import_scope)) as Tensor;
            // The boolean tensor for loop termination condition.
            _pivot = g.as_graph_element(ops.prepend_name_scope(context_def.PivotName, import_scope)) as Tensor;
            // The list of exit tensors for loop variables.
            _loop_exits = new List<Tensor>();
            foreach (var (i, exit_name) in enumerate(context_def.LoopExitNames))
                _loop_exits.Add(g.as_graph_element(ops.prepend_name_scope(exit_name, import_scope)) as Tensor);
            // The list of enter tensors for loop variables.
            _loop_enters = new List<Tensor>();
            foreach (var (i, enter_name) in enumerate(context_def.LoopEnterNames))
                _loop_enters.Add(g.as_graph_element(ops.prepend_name_scope(enter_name, import_scope)) as Tensor);

            __init__(values_def: context_def.ValuesDef, import_scope: import_scope);
        }

        /// <summary>
        /// Add the loop termination condition and body to the graph.
        /// </summary>
        internal Tensor[] BuildLoop<TItem>(Func<LoopVar<TItem>, Tensor> pred, 
            Func<LoopVar<TItem>, LoopVar<TItem>> body,
            LoopVar<TItem> loop_vars,
            TensorShape[] shape_invariants,
            bool return_same_structure)
        {
            // Keep original_loop_vars to identify which are TensorArrays
            var original_loop_vars = loop_vars;
            // Convert TensorArrays to their flow variables
            var loop_vars_tensors = nest.flatten2(loop_vars)
                .Select(x => _convert_tensorarray_to_flow(x))
                .ToArray();

            if (shape_invariants == null)
                shape_invariants = loop_vars_tensors
                    .Select(x => _get_shape_invariant(x as Tensor))
                    .ToArray();

            Enter();
            var(original_body_result, exit_vars) = _BuildLoop(
                pred, body, original_loop_vars, loop_vars_tensors, shape_invariants);
            Exit();

            var flat_result = original_body_result;

            var exit_vars_with_tensor_arrays = _convert_flows_to_tensorarrays(flat_result, exit_vars);
            var packed_exit_vars = nest.pack_sequence_as(
                structure: original_body_result,
                flat_sequence: exit_vars_with_tensor_arrays);

            return packed_exit_vars as Tensor[];
        }

        private Tensor _convert_tensorarray_to_flow(object tensor_or_tensor_array)
        {
            if (tensor_or_tensor_array is TensorArray tensor_array)
                return tensor_array.flow;
            else if (tensor_or_tensor_array is Tensor tensor)
                return tensor;

            throw new NotImplementedException("_convert_tensorarray_to_flow");
        }

        private TensorShape _get_shape_invariant(Tensor var, int[] shape = null)
        {
            return var.TensorShape;
        }

        /// <summary>
        /// Add the loop termination condition and body to the graph.
        /// </summary>
        /// <typeparam name="TItem"></typeparam>
        /// <param name="pred"></param>
        /// <param name="body"></param>
        /// <param name="original_loop_vars"></param>
        /// <param name="loop_vars"></param>
        /// <param name="shape_invariants"></param>
        /// <returns></returns>
        private (Tensor[], Tensor[]) _BuildLoop<TItem>(Func<LoopVar<TItem>, Tensor> pred,
            Func<LoopVar<TItem>, LoopVar<TItem>> body,
            LoopVar<TItem> original_loop_vars,
            Tensor[] loop_vars,
            TensorShape[] shape_invariants)
        {
            var flat_loop_vars = nest.flatten2(original_loop_vars)
                .Select(x => (ITensorOrTensorArray)x)
                .ToArray();

            // Let the context know the loop variables so the loop variables
            // would be added in the outer contexts properly.
            _InitializeValues(loop_vars);
            var real_vars = loop_vars;
            Tensor[] enter_vars = null;
            tf_with(ops.control_dependencies(null), delegate
            {
                enter_vars = real_vars.Select(x => _Enter(x,
                    _name,
                    is_constant: false,
                    parallel_iterations: _parallel_iterations,
                    use_input_shape: shape_invariants == null))
                .ToArray();

                foreach (var x in enter_vars)
                {
                    x.graph.prevent_feeding(x);
                    if (_outer_context != null)
                        _outer_context.AddInnerOp(x.op);
                }
            });

            // Finds the closest enclosing non-None control pivot.
            var outer_context = _outer_context;
            object control_pivot = null;
            while (outer_context != null && control_pivot == null)
            {

            }

            if (control_pivot != null)
            {

            }

            _SetShapeInvariants(real_vars, enter_vars, shape_invariants);

            // Fix the control inputs and control flow context of these enter ops.
            _FixControlInputsAndContext(enter_vars);
            _InitializeValues(enter_vars);
            _loop_enters = enter_vars.ToList();

            var merge_vars = enter_vars
                .Select(x => merge(new[] { x, x }))
                .ToArray();

            _pivot_for_pred = merge_vars[0];

            // Build the graph for pred.
            var merge_vars_with_tensor_arrays = _convert_flows_to_tensorarrays(flat_loop_vars, merge_vars);
            //var packed_vars = nest.pack_sequence_as(original_loop_vars, merge_vars_with_tensor_arrays, expand_composites: true);
            var packed_vars = new LoopVar<TItem>((Tensor)merge_vars_with_tensor_arrays[0],
                (TItem)(object)new BodyItemInRnnWhileLoop((Tensor)merge_vars_with_tensor_arrays[1],
                new[] { (TensorArray)merge_vars_with_tensor_arrays[2] },
                (Tensor)merge_vars_with_tensor_arrays[3]));
            var pp = pred(packed_vars);
            var c = ops.convert_to_tensor(pp);
            _pivot = gen_control_flow_ops.loop_cond(c, name: "LoopCond");
            var switch_vars = merge_vars.Select(x => _SwitchRefOrTensor(x, _pivot))
                .ToArray();

            // Build the graph for body.
            var vars_for_body = switch_vars.Select(x => _Identity(x[1])).ToArray();
            _pivot_for_body = vars_for_body[0];
            // Convert TensorArray flow variables inside the context back into
            // their associated TensorArrays for calling the body.
            var vars_for_body_with_tensor_arrays = _convert_flows_to_tensorarrays(flat_loop_vars, vars_for_body);
            var packed_vars_for_body = nest.pack_sequence_as2(original_loop_vars, vars_for_body_with_tensor_arrays);
            var pre_summaries = ops.get_collection(tf.GraphKeys._SUMMARY_COLLECTION);
            var body_result = body(packed_vars_for_body);
            var post_summaries = ops.get_collection(tf.GraphKeys._SUMMARY_COLLECTION);

            // Store body_result to keep track of TensorArrays returned by body
            var original_body_result = new[] { body_result };
            // Convert TensorArrays returned by body into their flow variables
            var result = new[] { body_result };

            var next_vars = new List<Tensor>();
            //foreach (var (m, v) in zip(merge_vars, result))
                //next_vars.Add(_AddNextAndBackEdge(m, v));

            // Add the exit ops.
            var exit_vars = switch_vars.Select(x => exit(x[0])).ToList();
            _loop_exits = exit_vars;

            // Exit the loop.
            // ExitResult(exit_vars);
            return (null, exit_vars.ToArray());
        }

        private void _FixControlInputsAndContext(Tensor[] enters)
        {
            var graph = ops.get_default_graph();
            foreach(var x in enters)
            {
                var inp_op = x.op.inputs[0].op;
                var control_inputs = graph._control_dependencies_for_inputs(new[] { inp_op });
                var outer_control_inputs = new List<Operation>();
                foreach(Operation op in control_inputs)
                {
                    // We need to keep control inputs that are in any ancestor
                    // ControlFlowContext, and within outer WhileContext.
                    var keep_as_control_input = true;
                    var op_ctxt = control_flow_util.GetOutputContext(op);
                    var outer_ctxt = outer_context;
                    throw new NotImplementedException("");
                }
                // op for op in control_inputs if self._IsInOuterContext(op)
                /*var outer_control_inputs = control_inputs.Where(x => _IsInOuterContext(x.op))
                    .Select(x => x.op)
                    .ToArray();*/
                x.op._set_control_flow_context(this);
                x.op._add_control_inputs(outer_control_inputs.ToArray());
                graph._record_op_seen_by_control_dependencies(x.op);
            }
        }

        private void _InitializeValues(Tensor[] values)
        {
            _values = new HashSet<string>();
            foreach(var x in values)
                _values.Add(x.name);
        }

        protected override void _AddOpInternal(Operation op)
        {
            Operation[] external_inputs = new Operation[0];
            if (op == null)
            {
                throw new NotImplementedException("");
            }
            else
            {
                foreach (var index in range(len(op.inputs)))
                {
                    var x = op.inputs[index];
                    var real_x = AddValue(x);
                    if (real_x != x)
                        op._update_input(index, real_x);
                }

                // Remove any external control dependency on this op.
                (_, external_inputs) = _RemoveExternalControlEdges(op);
                // Add a control dependency to prevent loop invariants from
                // enabling ops that should not be executed.
                _MaybeAddControlDependency(op);
                foreach (Tensor x in op.outputs)
                    _values.Add(x.name);
            }

            if (external_inputs.Length > 0)
            {
                throw new NotImplementedException("external_inputs.Length > 0");
            }

            if (_outer_context != null || !IsLoopExit(op))
                foreach (Tensor x in op.outputs)
                    op.graph.prevent_feeding(x);

            if (_outer_context != null)
                _outer_context.AddInnerOp(op);
        }

        protected void _MaybeAddControlDependency(Operation op)
        {
            // Determines if `op` needs a control dependency.
            Func<Operation, bool> _IsOpFree = (op1) =>
            {
                if (op1.control_inputs.Length > 0)
                    return false;

                if (op1.type == "SymbolicGradient")
                    return true;

                foreach (Tensor x in op1.inputs)
                    if (!control_flow_util.IsLoopConstantEnter(x.op))
                        return false;

                return true;
            };

            if (_IsOpFree(op))
                op._add_control_input(GetControlPivot().op);
        }

        private Tensor GetControlPivot()
        {
            if (_pivot_for_body != null)
                return _pivot_for_body;
            return _pivot_for_pred;
        }

        public override void AddOp(Operation op)
        {
            _AddOpInternal(op);
        }

        public override Tensor AddValue(Tensor val)
        {
            var result = val;
            var new_value = !_values.Contains(val.name);
            new_value &= val.op._get_control_flow_context() != this;
            if (new_value)
            {
                _values.Add(val.name);

                // If we are in a grad context and val is from its forward context,
                // use GetRealValue(), which adds the logic to save the history of
                // val in forward.
                var grad_ctxt = ops.get_default_graph()._get_control_flow_context();
                if(grad_ctxt != null)
                {
                    grad_ctxt = grad_ctxt.GetWhileContext();
                    if (grad_ctxt.grad_state != null)
                    {
                        throw new NotImplementedException("");
                    }
                }

                if (_outer_context != null)
                {
                    result = _outer_context.AddValue(val);
                }

                // Create an Enter to make `result` known to this loop context.
                Tensor enter = null;
                tf_with(ops.control_dependencies(new ITensorOrOperation[0]), delegate
                {
                    enter = _Enter(
                        result,
                        _name,
                        is_constant: true,
                        parallel_iterations: _parallel_iterations);
                    enter.graph.prevent_feeding(enter);
                    if (_outer_context != null)
                        _outer_context.AddInnerOp(enter.op);
                });

                // Fix the control inputs and control flow context of these enter ops.
                _FixControlInputsAndContext(new[] { enter });
                // Add `enter` in this context.
                _values.Add(enter.name);
                _external_values[val.name] = enter;
                result = enter;
            }
            else
            {
                var actual_val = _external_values.ContainsKey(val.name) ? _external_values[val.name] : null;
                if (actual_val != null)
                    result = actual_val as Tensor;
            }

            return result;
        }

        public override WhileContext GetWhileContext()
        {
            return this;
        }

        public WhileContext from_proto(WhileContextDef proto, string import_scope)
        {
            var ret = new WhileContext(context_def: proto, import_scope: import_scope);

            ret.Enter();
            foreach (var nested_def in proto.NestedContexts)
                from_control_flow_context_def(nested_def, import_scope: import_scope);
            ret.Exit();
            return ret;
        }

        public object to_proto()
        {
            throw new NotImplementedException();
        }
    }
}
