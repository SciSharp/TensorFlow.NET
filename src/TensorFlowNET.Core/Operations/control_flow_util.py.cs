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
using System.Linq;
using Tensorflow.Functions;
using Tensorflow.Graphs;
using Tensorflow.Operations;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class control_flow_util
    {
        public static readonly bool ENABLE_CONTROL_FLOW_V2 = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TF_ENABLE_CONTROL_FLOW_V2")) && Environment.GetEnvironmentVariable("TF_ENABLE_CONTROL_FLOW_V2") != "0" ||
                              (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TF_ENABLE_CONTROL_FLOW_V2")) && Environment.GetEnvironmentVariable("TF_ENABLE_CONTROL_FLOW_V2") != "0") ||
                              (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TF_ENABLE_COND_V2")) && Environment.GetEnvironmentVariable("TF_ENABLE_COND_V2") != "0") ||
                              (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TF_ENABLE_WHILE_V2")) && Environment.GetEnvironmentVariable("TF_ENABLE_WHILE_V2") != "0") ||
                              (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TF_ENABLE_TENSOR_ARRAY_V2")) && Environment.GetEnvironmentVariable("TF_ENABLE_TENSOR_ARRAY_V2") != "0");
        /// <summary>
        /// Return true if `op` is an Exit.
        /// </summary>
        /// <param name="op"></param>
        /// <returns></returns>
        public static bool IsLoopExit(Operation op)
        {
            return op.type == "Exit" || op.type == "RefExit";
        }

        /// <summary>
        /// Returns true if `op` is an Enter.
        /// </summary>
        /// <param name="op"></param>
        /// <returns></returns>
        public static bool IsLoopEnter(Operation op)
        {
            return op.type == "Enter" || op.type == "RefEnter";
        }

        /// <summary>
        /// Return true iff op is a loop invariant.
        /// </summary>
        /// <param name="op"></param>
        /// <returns></returns>
        public static bool IsLoopConstantEnter(Operation op)
        {
            return IsLoopEnter(op) && op.get_attr<bool>("is_constant");
        }

        /// <summary>
        /// Return true if `op` is a Switch.
        /// </summary>
        /// <param name="op"></param>
        /// <returns></returns>
        public static bool IsSwitch(Operation op)
        {
            return op.type == "Switch" || op.type == "RefSwitch";
        }

        public static WhileContext GetWhileContext(Operation op)
            => op.GetWhileContext();

        public static bool IsCondSwitch(Operation op)
        {
            if (!IsSwitch(op))
                return false;
            if (op.outputs == null || op.outputs.Length == 0)
                return false;

            // Switch nodes are not part of the cond control flow context that they
            // represent, so consider the consumers of its outputs to determine if it is
            // cond switch or not. A switch is a cond switch iff all its consumers are in
            // cond contexts.
            var is_cond_switch = true;
            foreach (var o in op.outputs)
            {
                foreach (var c in o.consumers())
                {
                    var ctxt = c._get_control_flow_context();
                    if (IsLoopEnter(c))
                        ctxt = ctxt.outer_context;
                    is_cond_switch = is_cond_switch && (ctxt != null && ctxt.IsCondContext());
                }
            }

            return is_cond_switch;
        }

        public static bool IsLoopSwitch(Operation op)
        {
            if (IsSwitch(op))
            {
                var ctxt = op._get_control_flow_context();
                return ctxt != null && ctxt.IsWhileContext() && !IsCondSwitch(op);
            }
            return false;
        }

        /// <summary>
        /// Return the control flow context for the output of an op.
        /// </summary>
        public static ControlFlowContext GetOutputContext(Operation op)
        {
            var ctxt = op._get_control_flow_context();
            // Exit nodes usually have a control flow context, except in the case where the
            // exit node was imported via import_graph_def (in which case no nodes have
            // control flow contexts).
            if (ctxt != null && IsLoopExit(op))
                ctxt = ctxt.outer_context;
            return ctxt;
        }

        public static void CheckInputFromValidContext(Operation op, Operation input_op)
        {
            var op_ctxt = op._get_control_flow_context();
            var input_ctxt = GetOutputContext(input_op);
            var valid = false;
            if (input_ctxt == null)
                valid = true;
            else if (op_ctxt == input_ctxt)
                valid = true;
            else
            {
                var while_ctxt = GetContainingWhileContext(op_ctxt);
                var input_while_ctxt = GetContainingWhileContext(input_ctxt);

                if (while_ctxt == null)
                {
                    // Neither op nor input_op is in a while loop, but one or both are in
                    // conds. We allow this, although execution will fail if the branch
                    // corresponding to input_op's cond context isn't taken.
                    if (input_while_ctxt == null)
                        valid = true;
                    // Invalid if op isn't in a while loop and input_op is. Unless...
                    if (IsLoopEnter(op))
                        // WhileContext._BuildLoop clears context for Enter nodes.
                        valid = true;
                    if (IsSwitch(op))
                        // CondContext.AddValue clears context for Switch nodes.
                        valid = true;
                }
                else if (IsContainingContext(while_ctxt, input_while_ctxt))
                {
                    // input_op is in a while loop which contains op's while loop (or not in a
                    // while loop at all).
                    valid = true;
                }
                else if (while_ctxt.grad_state != null &&
                    IsContainingContext(while_ctxt.grad_state.forward_context,
                              input_while_ctxt))
                {
                    valid = true;
                }
                else
                    throw new NotImplementedException("CheckInputFromValidContext");
            }

            if (!valid)
            {
                throw new NotImplementedException("CheckInputFromValidContext");
            }
        }

        public static Operation GetLoopConstantEnter(Tensor value)
        {
            var id_ops = new string[] { "Switch", "RefSwitch", "Identity", "RefIdentity" };
            var op = value.op;
            while (id_ops.Contains(op.type))
                op = op.inputs[0].op;
            return IsLoopConstantEnter(op) ? op : null;
        }

        public static bool IsContainingContext(WhileContext ctxt, WhileContext maybe_containing_ctxt)
        {
            while (ctxt != maybe_containing_ctxt)
            {
                if (ctxt == null)
                    return false;
                ctxt = ctxt.outer_context as WhileContext;
            }
            return true;
        }

        public static WhileContext GetContainingWhileContext(ControlFlowContext ctxt, ControlFlowContext stop_ctxt = null)
        {
            while (ctxt != null)
            {
                if (ctxt.IsWhileContext() || ctxt == stop_ctxt)
                    return ctxt as WhileContext;
                ctxt = ctxt.outer_context;
            }
            return null;
        }

        public static bool EnableControlFlowV2(Graph graph)
        {
            return ENABLE_CONTROL_FLOW_V2 || graph.building_function && (graph is not FuncGraph func || func.captures.Length == 0);
            
        }

        public static string create_new_tf_function(FuncGraph func_graph)
        {
            var func = new EagerDefinedFunction(func_graph.Name, func_graph, func_graph.Inputs, func_graph.Outputs, new Dictionary<string, AttrValue>());
            func.AddToGraph(func_graph);
            return func_graph.Name;
        }

        public static (Operation, Tensor[]) get_op_and_outputs(Tensor[] inputs)
        {
            if(inputs.Length == 0)
            {
                return (null, new Tensor[0]);
            }
            else
            {
                return (inputs[0], inputs);
            }
        }

        public static Tensor[] run_as_function_for_tape_gradients(Func<Tensor[], Tensor[]> make_op, Tensor[] inputs)
        {
            if(gradients_util.PossibleTapeGradientTypes(inputs) == gradients_util.POSSIBLE_GRADIENT_TYPES_HIGHER_ORDER
                && !(ops.get_default_graph().building_function))
            {
                throw new NotImplementedException();
            }
            else
            {
                return make_op(inputs);
            }
        }

        public static string unique_fn_name(string scope, string name)
        {
            return $"{scope}{name}_{ops.uid()}".Replace("/", "_");
        }

        public static bool output_all_intermediates()
        {
            if (in_defun())
            {
                return false;
            }
            if(tf.Context.FunctionCallOptions.ExecutorType == "SINGLE_THREADED_EXECUTOR")
            {
                return false;
            }
            // TODO(Rinne): check this after refactoring keras building.
            return false;
        }

        public static bool in_defun()
        {
            if (tf.Context.executing_eagerly())
            {
                return false;
            }

            var graph = ops.get_default_graph();
            // TODO(Rinne): CondBranchFuncGraph, WhileBodyFuncGraph, WhileCondFuncGraph
            return graph is FuncGraph;
        }
    }
}
