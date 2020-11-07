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

namespace Tensorflow.Gradients
{
    /// <summary>
    /// Gradients for operators defined in control_flow_ops.py.cs
    /// </summary>
    [RegisterGradient("control_flow_grad")]
    public class control_flow_grad
    {
        /// <summary>
        /// Gradients for a Switch op is calculated using a Merge op.
        ///
        /// If the switch is a loop switch, it will be visited twice. We create
        /// the merge on the first visit, and update the other input of the merge
        /// on the second visit. A next_iteration is also added on second visit.
        /// </summary>
        /// <returns></returns>
        [RegisterGradient("Switch")]
        public static Tensor[] _SwitchGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var graph = ops.get_default_graph();
            var op_ctxt = op._get_control_flow_context();
            var grad_ctxt = graph._get_control_flow_context();
            switch (op_ctxt)
            {
                case WhileContext cwhile:
                    {
                        var merge_grad = grad_ctxt.grad_state.switch_map.get(op);
                        if (merge_grad != null)
                        {
                            if (grads[1] != null)
                                control_flow_ops._AddNextAndBackEdge(merge_grad, grads[1],
                                             enforce_shape_invariant: false);
                            return new Tensor[] { null, null };
                        }
                        else if (grads[0] != null)
                        {
                            merge_grad = merge(new[] { grads[0], grads[0] }, name: "b_switch")[0];
                            grad_ctxt.grad_state.switch_map[op] = merge_grad;
                            return new Tensor[] { merge_grad, null };
                        }
                        else
                            return new Tensor[] { null, null };
                    }
                case CondContext ccond:
                    {
                        var zero_grad = grads[1 - op_ctxt.branch];
                        // At this point, we have created zero_grad guarded by the right switch.
                        // Unfortunately, we may still get None here for not trainable data types.
                        if (zero_grad == null)
                        {
                            throw new NotImplementedException("_SwitchGrad CondContext zero_grad");
                        }

                        return new Tensor[]
                        {
                            merge(grads, name: "cond_grad")[0],
                            null
                        };
                    }
                default:
                    throw new NotImplementedException("_SwitchGrad WhileContext");
            }
            throw new NotImplementedException("_SwitchGrad");
        }

        /// <summary>
        /// Returns the value of an available element of `inputs`.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        internal static MergeOutput merge(Tensor[] inputs, string name = null)
        {
            return tf_with(ops.name_scope(name, "Merge", inputs), scope =>
            {
                name = scope;
                if (inputs.Count(x => x.dtype.is_ref_dtype()) == inputs.Length)
                    return gen_control_flow_ops.ref_merge(inputs, name: name);
                else
                    return gen_control_flow_ops.merge(inputs, name: name);
            });
        }

        /// <summary>
        /// Gradients for a Merge op are calculated using a Switch op.
        /// </summary>
        [RegisterGradient("Merge")]
        public static Tensor[] _MergeGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var input_op = op.inputs[0].op;
            var graph = ops.get_default_graph();
            var op_ctxt = control_flow_util.GetOutputContext(input_op);
            var grad_ctxt = graph._get_control_flow_context();
            switch (op_ctxt)
            {
                case WhileContext cwhile:
                    {
                        return control_flow_ops._SwitchRefOrTensor(grad, grad_ctxt.pivot);
                    }
                case CondContext ccond:
                    {
                        var pred = ccond.pred;
                        if (grad_ctxt != null && grad_ctxt.grad_state != null)
                        {
                            //# This Merge node is part of a cond within a loop.
                            //# The backprop needs to have the value of this predicate for every
                            //# iteration. So we must have its values accumulated in the forward, and
                            //# use the accumulated values as the predicate for this backprop switch.
                            var grad_state = grad_ctxt.grad_state;
                            var real_pred = grad_state.history_map[pred.name] as Tensor;
                            if (real_pred == null)
                            {
                                //# Remember the value of pred for every iteration.
                                grad_ctxt = grad_state.grad_context;
                                grad_ctxt.Exit();
                                var history_pred = grad_state.AddForwardAccumulator(pred);
                                grad_ctxt.Enter();

                                //# Add the stack pop op. If pred.op is in a (outer) CondContext,
                                //# the stack pop will be guarded with a switch.
                                real_pred = grad_state.AddBackpropAccumulatedValue(history_pred, pred);
                                grad_state.history_map[pred.name] = real_pred;
                            }
                            pred = real_pred;
                        }
                        var results = control_flow_ops._SwitchRefOrTensor(grad, pred, name: "cond_grad");
                        return results;
                    }
                default:
                    {
                        var num_inputs = op.inputs.Length;
                        var cond = new Tensor[num_inputs];
                        for (int i = 0; i < num_inputs; i++)
                            cond[i] = math_ops.equal(op.outputs[1], i);
                        var result = cond.Select(t => control_flow_ops._SwitchRefOrTensor(grad, t)[1]).ToArray();
                        return result;
                    }
            }

        }

        [RegisterGradient("RefMerge")]
        public static Tensor[] _RefMergeGrad(Operation op, Tensor[] grads)
        {
            return _MergeGrad(op, grads);
        }

        /// <summary>
        /// Gradients for an exit op are calculated using an Enter op.
        /// </summary>
        [RegisterGradient("Exit")]
        public static Tensor[] _ExitGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var graph = ops.get_default_graph();
            var op_ctxt = op._get_control_flow_context();
            var grad_ctxt = graph._get_control_flow_context() as WhileContext;
            // The flag `back_prop` is set by users to suppress gradient
            // computation for this loop. If the attribute `back_prop` is false,
            // no gradient computation.
            if (!grad_ctxt.back_prop)
                return null;

            if (op_ctxt.grad_state != null)
                throw new TypeError("Second-order gradient for while loops not supported.");

            grad_ctxt.AddName(grad.name);

            grad_ctxt.Enter();
            var result = control_flow_ops._Enter(
                  grad, grad_ctxt.Name, is_constant: false,
                  parallel_iterations: grad_ctxt.parallel_iterations,
                 name: "b_exit");

            grad_ctxt.loop_enters.append(result);
            grad_ctxt.Exit();
            return new[] { result };
        }

        /// <summary>
        /// A forward next_iteration is translated into a backprop identity.
        ///
        ///  Note that the backprop next_iteration is added in switch grad.
        /// </summary>
        [RegisterGradient("NextIteration")]
        public static Tensor[] _NextIterationGrad(Operation op, Tensor[] grads)
        {
            return grads;
        }

        [RegisterGradient("RefNextIteration")]
        public static Tensor[] _RefNextIterationGrad(Operation op, Tensor[] grads)
        {
            return grads;
        }

        /// <summary>
        /// Gradients for an Enter are calculated using an Exit op.
        /// 
        ///  For loop variables, grad is the gradient so just add an exit.
        ///  For loop invariants, we need to add an accumulator loop.
        /// </summary>
        [RegisterGradient("Enter")]
        public static Tensor[] _EnterGrad(Operation op, Tensor[] grads)
        {
            Tensor result = null;
            var grad = grads[0];
            var graph = ops.get_default_graph();
            var grad_ctxt = graph._get_control_flow_context() as WhileContext;
            if (!grad_ctxt.back_prop)
                // Skip gradient computation, if the attribute `back_prop` is false.
                return grads;
            if (grad_ctxt.grad_state == null)
                // Pass the gradient through if we are not in a gradient while context.
                return grads;
            if (op.get_attr<bool>("is_constant"))
            {
                // Add a gradient accumulator for each loop invariant.
                result = grad_ctxt.AddBackpropAccumulator(op, grad);
            }
            else
            {
                result = control_flow_ops.exit(grad);
                grad_ctxt.loop_exits.append(result);
                grad_ctxt.ExitResult(new[] { result });
            }

            return new Tensor[] { result };
        }


        [RegisterGradient("RefEnter")]
        public Tensor[] _RefEnterGrad(Tensor op, Tensor[] grad)
        {
            return _EnterGrad(op, grad);
        }

        /// <summary>
        /// Stop backprop for the predicate of a while loop.
        /// </summary>
        [RegisterGradient("LoopCond")]
        public Tensor[] _LoopCondGrad(Tensor op, Tensor[] grad)
        {
            return null;
        }

    }
}
