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
using util = Tensorflow.control_flow_util;

namespace Tensorflow.Operations.ControlFlows
{
    /// <summary>
    /// Maintain the mapping from the loops to their grad states.
    /// </summary>
    public class ControlFlowState
    {
        Dictionary<ControlFlowContext, GradLoopState> _map;
        //class ControlFlowState(object):
        //  """Maintain the mapping from the loops to their grad states."""

        //  def __init__(self):
        //    self._map = {}  # maps forward loop context to GradLoopState

        //  def GetGradState(self, op, before):
        //    """Return the grad state for this op if it's in a forward loop context."""
        //    if before and util.IsLoopExit(op):
        //      forward_ctxt = op._get_control_flow_context()
        //      forward_ctxt = forward_ctxt.outer_context
        //      if forward_ctxt:
        //        forward_ctxt = forward_ctxt.GetWhileContext()
        //    else:
        //      forward_ctxt = _GetWhileContext(op)
        //    if forward_ctxt:
        //      return self._map.get(forward_ctxt)
        //    return None

        public ControlFlowState()
        {
            _map = new Dictionary<ControlFlowContext, GradLoopState>();
        }

        /// <summary>
        /// Return the grad state for this op if it's in a forward loop context.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="before"></param>
        /// <returns></returns>
        public GradLoopState GetGradState(Operation op, bool before)
        {
            ControlFlowContext forward_ctxt = null;
            if (before && util.IsLoopExit(op))
            {
                forward_ctxt = op._get_control_flow_context();
                forward_ctxt = forward_ctxt.outer_context;
                if (forward_ctxt != null)
                    forward_ctxt = forward_ctxt.GetWhileContext();
            }
            else
                forward_ctxt = util.GetWhileContext(op);
            if (forward_ctxt != null)
                return _map.get(forward_ctxt);
            return null;
        }

        public Tensor[] ProcessUnusedLoopExits(Dictionary<string, int> pending_count, List<Operation> to_ops_set)
        {
            var loop_exits = new List<Tensor>();
            foreach (var grad_state in _map.Values)
            {
                foreach (var y in grad_state.forward_loop_exits)
                {
                    if (!pending_count.ContainsKey(y.op.name))
                    {
                        grad_state.pending_exits_count -= 1;
                        if (!to_ops_set.Contains(y.op))
                            grad_state.unused_exits.append(y);
                        if (grad_state.pending_exits_count == 0)
                            loop_exits.extend(grad_state.unused_exits);
                    }
                }

                foreach (var y in grad_state.forward_context.loop_enters)
                {
                    if (!pending_count.ContainsKey(y.op.name))
                        pending_count[y.op.name] = 1;
                }
            }

            return loop_exits.ToArray();
        }

        public void EnterGradWhileContext(Operation op, bool before)
        {
            var grad_state = GetGradState(op, before);
            if (grad_state != null)
                grad_state.grad_context.Enter();
        }

        public void ExitGradWhileContext(Operation op, bool before)
        {
            var grad_state = GetGradState(op, before);
            if (grad_state != null)
                grad_state.grad_context.Exit();
        }

        //  def AddWhileContext(self, op, between_op_list, between_ops):
        //    """Add the grad state for the while loop that op belongs to.

        //    Note that op is an Exit, and this method must be called in
        //    the control flow context where gradients() is called.

        //    Note that this method modifies `between_op_list` and `between_ops`.
        //    """
        //    forward_ctxt = _GetWhileContext(op)
        //    grad_state = self._map.get(forward_ctxt)
        //    if grad_state is None:
        //      # This is a new while loop so create a grad state for it.
        //      outer_forward_ctxt = forward_ctxt.outer_context
        //      if outer_forward_ctxt:
        //        outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
        //      outer_grad_state = None
        //      if outer_forward_ctxt:
        //        outer_grad_state = self._map.get(outer_forward_ctxt)
        //      grad_state = GradLoopState(forward_ctxt, outer_grad_state)
        //      self._map[forward_ctxt] = grad_state

        //      # We need to include all exits of a loop for backprop.
        //      for loop_exit in grad_state.forward_loop_exits:
        //        if loop_exit.op not in between_ops:
        //          between_ops.add(loop_exit.op)
        //          between_op_list.append(loop_exit.op)
        public void AddWhileContext(Operation op, List<Operation> between_op_list, List<Operation> between_ops)
        {
            var forward_ctxt = op.GetWhileContext();
            var grad_state = _map.ContainsKey(forward_ctxt) ? _map[forward_ctxt] : null;
            if (grad_state == null)
            {
                GradLoopState outer_grad_state = null;
                var outer_forward_ctxt = forward_ctxt.outer_context;
                if (outer_forward_ctxt != null)
                    outer_forward_ctxt = outer_forward_ctxt.GetWhileContext();
                if (outer_forward_ctxt != null)
                    outer_grad_state = _map[outer_forward_ctxt];
                grad_state = new GradLoopState(forward_ctxt, outer_grad_state);
                _map[forward_ctxt] = grad_state;

                // We need to include all exits of a loop for backprop.
                foreach (var loop_exit in grad_state.forward_loop_exits)
                {
                    if (!between_ops.Contains(loop_exit.op))
                    {
                        between_ops.add(loop_exit.op);
                        between_op_list.append(loop_exit.op);
                    }
                }
            }
        }

        //  def ZerosLikeForExit(self, val):
        //    """Create zeros_like gradient for a loop exit.

        //    If the result of a loop variable is not used but is involved in
        //    computing the result of some needed loop variable, we create a
        //    zero-valued tensor that is fed as gradient for the Exit node of that
        //    loop variable. Note that val.op is an Exit, and this method must be
        //    called in the control flow context where gradients() is called.

        //    Args:
        //      val: The output tensor of an Exit op.

        //    Returns:
        //      A zero tensor of the same shape of val.
        //    """
        //    val_shape = val.get_shape()
        //    forward_ctxt = val.op._get_control_flow_context()
        //    outer_forward_ctxt = forward_ctxt.outer_context
        //    if outer_forward_ctxt:
        //      outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
        //    outer_grad_state = None
        //    if outer_forward_ctxt:
        //      outer_grad_state = self._map.get(outer_forward_ctxt)
        //    if outer_grad_state:
        //      # This is a nested loop.
        //      if val_shape.is_fully_defined():
        //        # If the shape is known statically, just create a zero tensor
        //        # with the right shape in the right context.
        //        outer_grad_state.grad_context.Enter()
        //        result = array_ops.zeros(val_shape.dims, val.dtype)
        //        outer_grad_state.grad_context.Exit()
        //      else:
        //        # Only the shape of value is needed for backprop.
        //        forward_ctxt.outer_context.Enter()
        //        shape = array_ops.shape_internal(val, optimize=False)
        //        forward_ctxt.outer_context.Exit()
        //        # Save the shape to a stack.
        //        history_shape = outer_grad_state.AddForwardAccumulator(shape)
        //        # Get the shape back from the stack.
        //        outer_grad_ctxt = outer_grad_state.grad_context
        //        outer_grad_ctxt.Enter()
        //        real_shape = outer_grad_state.AddBackpropAccumulatedValue(
        //            history_shape, shape)
        //        result = array_ops.zeros(real_shape, val.dtype)
        //        outer_grad_ctxt.Exit()
        //    else:
        //      # This is not a nested loop.
        //      if val_shape.is_fully_defined():
        //        # If the shape is known statically, just create a zero tensor
        //        # with the right shape.
        //        result = array_ops.zeros(val_shape.dims, val.dtype)
        //      else:
        //        result = array_ops.zeros_like(val, optimize=False)
        //    return result

        public Tensor ZerosLike(Operation op, int index)
        {
            if (util.IsLoopSwitch(op))
                return null;
            if (op.graph.building_function)
                return array_ops.zeros_like(op.outputs[index]);
            var dead_branch = util.IsSwitch(op);
            var forward_ctxt = util.GetWhileContext(op);
            var grad_state = _map.get(forward_ctxt);
            // op is not in a while loop that is part of gradients().
            if (grad_state == null)
                return ZerosLikeOutsideLoop(op, index);
            throw new NotImplementedException("ZerosLike");
        }

        public Tensor ZerosLikeOutsideLoop(Operation op, int index)
        {
            var val = op.outputs[index];
            if (!util.IsSwitch(op))
            {
                if (val.dtype == dtypes.resource)
                    throw new NotImplementedException("ZerosLikeOutsideLoop");
                /*return array_ops.zeros(
                  gen_resource_variable_ops.variable_shape(val),
                  dtype: default_gradient.get_zeros_dtype(val));*/
                return array_ops.zeros_like(val, optimize: false);
            }
            else
                throw new NotImplementedException("ZerosLikeOutsideLoop");
        }

        /// <summary>
        /// Create zeros_like gradient for a loop exit.
        /// </summary>
        /// <param name="val"></param>
        /// <returns></returns>
        public Tensor ZerosLikeForExit(Tensor val)
        {
            Tensor result = null;
            var val_shape = val.shape;
            var forward_ctxt = val.op._get_control_flow_context();
            var outer_forward_ctxt = forward_ctxt.outer_context;
            if (outer_forward_ctxt != null)
                outer_forward_ctxt = outer_forward_ctxt.GetWhileContext();
            GradLoopState outer_grad_state = null;
            if (outer_forward_ctxt != null)
                outer_grad_state = _map.get(outer_forward_ctxt);
            // This is a nested loop.
            if (outer_grad_state != null)
            {
                throw new NotImplementedException("ZerosLikeForExit");
            }
            else
            {
                // If the shape is known statically, just create a zero tensor
                // with the right shape.
                if (val_shape.IsFullyDefined)
                    result = array_ops.zeros(val_shape.dims, val.dtype);
                else
                    result = array_ops.zeros_like(val, optimize: false);
            }
            return result;
        }

        public void PostProcessing()
        {
            foreach (var grad_state in _map.Values)
            {
                foreach (var b_merge in grad_state.switch_map.Values)
                {
                    if (b_merge.op.inputs[0] == b_merge.op.inputs[1])
                    {
                        Tensor next_grad_val = null;
                        // The value of this loop variable at iteration i+1 doesn't
                        // depend on its value at iteration i. So use zeros as the
                        // gradients for all iterations > 0.
                        var dtype = b_merge.op.inputs[0].dtype;
                        var shape = b_merge.op.inputs[0].shape;
                        if (shape.IsFullyDefined)
                        {
                            grad_state.grad_context.Enter();
                            // Create a zeros and use it for iterations > 0.
                            var grad_val = constant_op.constant(0, dtype: dtype, shape: shape);
                            next_grad_val = control_flow_ops._NextIteration(grad_val);
                            grad_state.grad_context.Exit();
                        }
                        else
                        {
                            throw new NotImplementedException("PostProcessing shape is not fully defined.");
                        }

                        b_merge.op._update_input(1, next_grad_val);
                    }
                }
            }
        }
    }
}
