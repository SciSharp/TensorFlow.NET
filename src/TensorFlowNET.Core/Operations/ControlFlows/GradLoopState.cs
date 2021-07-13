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
using System.Collections;
using System.Collections.Generic;
using static Tensorflow.Binding;
using util = Tensorflow.control_flow_util;

namespace Tensorflow.Operations.ControlFlows
{
    /// <summary>
    /// The state used for constructing the gradient graph for a while loop.
    /// </summary>
    public class GradLoopState
    {
        private WhileContext _grad_context = null;

        public WhileContext grad_context => _grad_context;

        //    # The loop counter added by AddBackpropLoopCounter. It is the value
        //    # of the loop counter for the current iteration.
        //    self._grad_index = None

        //    # A sync op for backprop.
        //    self._grad_sync = None

        //    # Information needed by backprop.
        private Hashtable _history_map = new Hashtable();
        public Hashtable history_map => _history_map;
        Dictionary<Operation, Tensor> _switch_map = new Dictionary<Operation, Tensor>();
        public Dictionary<Operation, Tensor> switch_map => _switch_map;

        /// <summary>
        /// The while loop context for forward.
        /// </summary>
        WhileContext _forward_context;
        public WhileContext forward_context => _forward_context;

        /// <summary>
        /// The grad loop state for the outer while loop.
        /// </summary>
        GradLoopState _outer_grad_state;
        public GradLoopState outer_grad_state => _outer_grad_state;

        Tensor _forward_index;
        public Tensor forward_index => _forward_index;
        Tensor _grad_index;

        Tensor[] _forward_loop_exits;
        /// <summary>
        /// The list of exits of the forward loop.
        /// </summary>
        public Tensor[] forward_loop_exits => _forward_loop_exits;

        List<Tensor> _deferred_exits;
        public List<Tensor> deferred_exits => _deferred_exits;

        List<Tensor> _unused_exits;
        public List<Tensor> unused_exits => _unused_exits;

        /// <summary>
        /// The number of exits we expect to see but haven't.
        /// </summary>
        public int pending_exits_count { get; set; }

        Operation _grad_sync;
        public Operation grad_sync
        {
            get
            {
                if (_grad_sync == null)
                {
                    tf_with(ops.control_dependencies(null), delegate
                    {
                        _grad_sync = gen_control_flow_ops.control_trigger(name: "b_sync");
                    });
                    _grad_sync._set_control_flow_context(_grad_context);
                    _grad_index.op._add_control_input(_grad_sync);
                    if (_grad_context.outer_context != null)
                        _grad_context.outer_context.AddInnerOp(_grad_sync);
                }
                return _grad_sync;
            }
        }

        public GradLoopState(WhileContext forward_ctxt, GradLoopState outer_grad_state_)
        {
            // Information needed by backprop.
            _unused_exits = new List<Tensor>();
            _deferred_exits = new List<Tensor>();
            _forward_loop_exits = list(forward_ctxt.loop_exits);
            pending_exits_count = len(forward_ctxt.loop_exits);

            _outer_grad_state = outer_grad_state_;

            ControlFlowContext outer_forward_ctxt = null;
            if (outer_grad_state_ != null)
                outer_forward_ctxt = outer_grad_state_.forward_context;

            // Add the forward loop counter.
            // with forward_ctxt._graph.as_default():
            Tensor cnt, forward_index;
            {
                if (outer_forward_ctxt != null)
                    outer_forward_ctxt.Enter();
                (cnt, forward_index) = forward_ctxt.AddForwardLoopCounter(outer_grad_state);
                if (outer_forward_ctxt != null)
                    outer_forward_ctxt.Exit();
            }
            _forward_context = forward_ctxt;
            _forward_index = forward_index;

            // Add the backprop WhileContext, and the backprop loop counter.
            if (outer_grad_state != null)
            {
                // This is a nested loop. Remember the iteration counts for each
                // execution of this inner loop.
                throw new NotImplementedException("GradLoopState");
            }
            else
            {
                if (outer_forward_ctxt != null)
                    outer_forward_ctxt.Enter();
                _grad_context = new WhileContext(
                  maximum_iterations: forward_ctxt.maximum_iterations,
                  parallel_iterations: forward_ctxt.parallel_iterations,
                  back_prop: forward_ctxt.back_prop,
                  swap_memory: forward_ctxt.swap_memory,
                  name: forward_ctxt.Name,
                  grad_state: this);
                _grad_index = _grad_context.AddBackpropLoopCounter(cnt, outer_grad_state);
                if (outer_forward_ctxt != null)
                    outer_forward_ctxt.Exit();
            }
        }

        /// <summary>
        /// Add an accumulator for each forward tensor that is needed in backprop.
        /// 
        ///    This is added to the forward loop at the first time when a tensor
        ///    in the forward loop is used by backprop gradient computation loop.
        ///    We create an accumulator that accumulates the value of tensor at each
        ///    iteration. Called in the control flow context where gradients() is called.
        ///
        ///    The pseudocode is:
        ///    ```
        ///      acc = stack();
        ///      while (_pivot) {
        ///        acc = stack_push(acc, value);
        ///      }
        ///   ```
        ///
        ///    We make sure that the stack push op in one iteration is executed before
        ///    next iteration. This is achieved by adding a control edge from
        ///    `forward_index.op.inputs[0].op` to the push op, and another control
        ///    edge from the push op to either `forward_index.op` or `forward_sync`.
        /// </summary>
        /// <param name="value"> The source tensor in forward that is to be accumulated.</param>
        /// <param name="dead_branch"> True iff the tensor is on a dead branch of a cond.</param>
        /// <returns>The stack that contains the accumulated history of the tensor.</returns>
        public Tensor AddForwardAccumulator(Tensor value, bool dead_branch = false)
        {
            _forward_index.graph.as_default();
            {
                var curr_ctxt = ops.get_default_graph()._get_control_flow_context();
                return tf_with(ops.control_dependencies(null), delegate
                {
                    Tensor acc = null;
                    Tensor push = null;
                    if (curr_ctxt != null)
                        curr_ctxt.Enter();
                    ops.colocate_with(value);
                    {
                        // We only need to pass maximum_iterations to the stack if
                        // we're inside an XLA context.
                        var max_size = constant_op.constant(-1, dtypes.int32);
                        acc = gen_data_flow_ops.stack_v2(
                            max_size: max_size, elem_type: value.dtype.as_base_dtype(), name: "f_acc");
                    }
                    if (curr_ctxt != null)
                        curr_ctxt.Exit();

                    // Make acc available in the forward context.
                    var enter_acc = forward_context.AddValue(acc);

                    // Add the stack_push op in the context of value.op.
                    var swap_enabled = forward_context.swap_memory;
                    var value_ctxt = util.GetOutputContext(value.op);
                    if (value_ctxt == forward_context)
                    {
                        // value is not nested in the forward context.
                        forward_context.Enter();
                        push = gen_data_flow_ops.stack_push_v2(enter_acc, value, swap_memory: swap_enabled);
                        forward_context.Exit();
                        // Protect stack push and order it before forward_index.
                        forward_index.op._add_control_input(push.op);
                    }
                    else
                    {
                        throw new NotImplementedException("AddForwardAccumulator");
                    }

                    // Order stack push after the successor of forward_index
                    var add_op = forward_index.op.inputs[0].op;
                    push.op._add_control_input(add_op);
                    return acc;
                });
            }
        }

        //    """Add the getter for an accumulated value in the grad context.
        //
        //    This is added to the backprop loop. Called in the grad context to
        //    get the value of an accumulated value. The stack pop op must be guarded
        //    by the pred of the controlling cond.
        //
        //    Args:
        //      history_value: The history (a stack) of a value.
        //      value: The value that is pushed onto the stack.
        //      dead_branch: True iff the tensor is on a dead branch of a cond.
        //
        //    Returns:
        //      The current value (the top of the stack).
        //    """

        public Tensor AddBackpropAccumulatedValue(Tensor history_value, Tensor value, bool dead_branch = false)
        {
            var history_ctxt = history_value.op._get_control_flow_context();
            // Find the cond context that controls history_value if any.
            CondContext cond_ctxt = null;
            Tensor pop = null;
            var value_ctxt = value.op._get_control_flow_context();
            while (value_ctxt != null && value_ctxt != history_ctxt)
            {
                if (value_ctxt is CondContext cc)
                    cond_ctxt = cc;
                value_ctxt = value_ctxt.outer_context;
            }
            tf_with(ops.control_dependencies(null), delegate
            {
                grad_context.Enter();
                if (cond_ctxt != null)
                {
                    throw new NotImplementedException("AddBackpropAccumulatedValue");
                }
                pop = gen_data_flow_ops.stack_pop_v2(history_value, value.dtype.as_base_dtype());
                pop.shape = value.shape;
                grad_context.Exit();
            });
            var parallel_iterations = grad_context.parallel_iterations;
            if (parallel_iterations > 1)
                // All pops are ordered after pivot_for_body and before grad_sync.
                grad_sync._add_control_input(pop.op);
            return pop;
        }

        /// <summary>
        /// Get the real value of `value`.
        /// </summary>
        /// <param name="value">A tensor to be captured.</param>
        /// <returns>The same tensor obtained from the saved history.</returns>
        public Tensor GetRealValue(Tensor value)
        {
            Tensor real_value = null;
            if (real_value == null)
            {
                var cur_value = value;
                var cur_grad_state = this;
                Tensor history_value = null;
                while (true)
                {
                    var enter_op = util.GetLoopConstantEnter(cur_value);
                    if (enter_op != null)
                    {
                        // Special case: cur_value comes from a constant Enter node.
                        cur_value = enter_op.inputs[0];
                        cur_grad_state = cur_grad_state.outer_grad_state;
                        if (cur_grad_state == null)
                        {
                            // We are now outside all nested loops for this gradient(),
                            // so `value` is a loop invariant and there is no need to
                            // save the history of value. Just make cur_value to enter
                            // the right control flow context.
                            real_value = _grad_context.AddValue(cur_value);
                            break;
                        }
                    }
                    else if (constant_op.is_constant(cur_value))
                    {
                        // We are now outside all nested loops for this gradient(),
                        // so `value` is a loop invariant and there is no need to
                        // save the history of value. Just make cur_value to enter
                        // the right control flow context.
                        real_value = constant_op.constant(
                            tensor_util.constant_value(cur_value), dtype: cur_value.dtype);
                        break;
                    }
                    else
                    {
                        // Record the history of this value in forward_ctxt.
                        _grad_context.Exit();
                        history_value = cur_grad_state.AddForwardAccumulator(cur_value);
                        _grad_context.Enter();
                        break;
                    }
                }

                if (real_value == null)
                {
                    // Add the stack pop op in the grad context.
                    real_value = cur_grad_state.AddBackpropAccumulatedValue(history_value, cur_value);
                    if (cur_grad_state != this)
                        real_value = _grad_context.AddValue(real_value);
                }
                _history_map[value.name] = real_value;
            }
            return real_value;
        }
    }
}
