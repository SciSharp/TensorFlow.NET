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
        internal Tensor[] BuildLoop<TItem>(Func<Tensor, TItem, Tensor> pred, 
            Func<Tensor, TItem, LoopVar<TItem>> body,
            LoopVar<TItem> loop_vars,
            TensorShape shape_invariants,
            bool return_same_structure)
        {
            // Keep original_loop_vars to identify which are TensorArrays
            var original_loop_vars = loop_vars;
            // Convert TensorArrays to their flow variables
            Enter();
            var(original_body_result, exit_vars) = _BuildLoop(
                pred, body, original_loop_vars, loop_vars, shape_invariants);
            Exit();

            var flat_result = original_body_result;

            var exit_vars_with_tensor_arrays = _convert_flows_to_tensorarrays(flat_result, exit_vars);
            var packed_exit_vars = nest.pack_sequence_as(
                structure: original_body_result,
                flat_sequence: exit_vars_with_tensor_arrays);

            return packed_exit_vars as Tensor[];
        }

        private Tensor _convert_tensorarray_to_flow<TItem>(TItem tensor_or_tensor_array)
        {
            if (tensor_or_tensor_array is TensorArray tensor_array)
                return tensor_array.flow;
            else if (tensor_or_tensor_array is Tensor tensor)
                return tensor;

            throw new NotImplementedException("_convert_tensorarray_to_flow");
        }

        private (Tensor[], Tensor[]) _BuildLoop<TItem>(Func<Tensor, TItem, Tensor> pred,
            Func<Tensor, TItem, LoopVar<TItem>> body,
            LoopVar<TItem> original_loop_vars,
            LoopVar<TItem> loop_vars,
            TensorShape shape_invariants)
        {
            var flat_loop_vars = original_loop_vars;

            // Convert TensorArrays to their flow variables
            var loop_vars_tensor = nest.map_structure(
                _convert_tensorarray_to_flow,
                nest.flatten2(loop_vars));

            // Let the context know the loop variables so the loop variables
            // would be added in the outer contexts properly.
            if (loop_vars is Tensor[] real_vars)
            {
                _InitializeValues(real_vars);
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
                while (outer_context != null)
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
                // var packed_vars = nest.pack_sequence_as(original_loop_vars, merge_vars_with_tensor_arrays);
                var c = ops.convert_to_tensor(pred(merge_vars_with_tensor_arrays[0], default(TItem)));
                _pivot = gen_control_flow_ops.loop_cond(c, name: "LoopCond");
                var switch_vars = merge_vars.Select(x => _SwitchRefOrTensor(x, _pivot))
                    .ToArray();

                // Build the graph for body.
                var vars_for_body = switch_vars.Select(x => _Identity(x[1])).ToArray();
                // Convert TensorArray flow variables inside the context back into
                // their associated TensorArrays for calling the body.
                var packed_vars_for_body = _convert_flows_to_tensorarrays(flat_loop_vars, vars_for_body);
                /*var body_result = body(packed_vars_for_body[0]);
                var post_summaries = ops.get_collection(tf.GraphKeys._SUMMARY_COLLECTION);

                // Store body_result to keep track of TensorArrays returned by body
                var original_body_result = new[] { body_result };
                // Convert TensorArrays returned by body into their flow variables
                var result = new[] { body_result };

                var next_vars = new List<Tensor>();
                foreach (var (m, v) in zip(merge_vars, result))
                    next_vars.Add(_AddNextAndBackEdge(m, v));

                // Add the exit ops.
                var exit_vars = switch_vars.Select(x => exit(x[0])).ToList();
                _loop_exits = exit_vars;

                // Exit the loop.
                // ExitResult(exit_vars);
                return (original_body_result, exit_vars.ToArray());*/
            }

            throw new NotImplementedException("");
        }

        private void _FixControlInputsAndContext(Tensor[] enters)
        {
            var graph = ops.get_default_graph();
            foreach(var e in enters)
            {
                var inp_op = e.op.inputs[0].op;
                var control_inputs = graph._control_dependencies_for_inputs(new[] { inp_op });
                // op for op in control_inputs if self._IsInOuterContext(op)
                var outer_control_inputs = control_inputs.Where(x => _IsInOuterContext(x.op))
                    .Select(x => x.op)
                    .ToArray();
                e.op._set_control_flow_context(this);
                e.op._add_control_inputs(outer_control_inputs);
                graph._record_op_seen_by_control_dependencies(e.op);
            }
        }

        private void _InitializeValues(Tensor[] values)
        {
            _values = new HashSet<string>();
            foreach(var x in values)
                _values.Add(x.name);
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
