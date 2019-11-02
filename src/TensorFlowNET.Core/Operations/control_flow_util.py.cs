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
using Tensorflow.Operations;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class control_flow_util
    {
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
            foreach(var o in op.outputs)
            {
                foreach(var c in o.consumers())
                {
                    var ctxt = c._get_control_flow_context();
                    if (IsLoopEnter(c))
                        ctxt = ctxt.outer_context;
                    is_cond_switch = is_cond_switch &&(ctxt != null && ctxt.IsCondContext());
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
                    throw new NotImplementedException("CheckInputFromValidContext");
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
            while(ctxt != maybe_containing_ctxt)
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
    }
}
