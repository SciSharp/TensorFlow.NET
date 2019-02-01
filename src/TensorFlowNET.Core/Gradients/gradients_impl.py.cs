using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace Tensorflow
{
    public class gradients_impl
    {
        public static void gradients(Tensor[] ys,
            Tensor[] xs, 
            Tensor[] grad_ys = null,
            string name = "gradients",
            bool colocate_gradients_with_ops = false,
            bool gate_gradients = false,
            int? aggregation_method = null)
        {
            _GradientsHelper(ys, xs, grad_ys, name, colocate_gradients_with_ops, gate_gradients);
        }

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
            var curr_graph = src_graph;

            if (grad_ys == null)
                grad_ys = new Tensor[ys.Length];

            var all = new List<Tensor>();
            all.AddRange(ys);
            all.AddRange(xs);
            all.AddRange(stop_gradients);
            all.AddRange(grad_ys);

            // Iterate over the collected ops.
            /**
             * grads: op => list of gradients received on each output endpoint of the
             * op.  The gradients for each endpoint are initially collected as a list.
             * When it is time to call the op's gradient function, for each endpoint we
             * aggregate the list of received gradients into a Add() Operation if there
             * is more than one.
             **/
            var grads = new Dictionary<string, Tensor[][]>();

            Python.with<ops.name_scope>(new ops.name_scope(name, "gradients", values: all), scope =>
            {
                string grad_scope = scope;
                // Get a uid for this call to gradients that can be used to help
                // cluster ops for compilation.
                var gradient_uid = ops.get_default_graph().unique_name("uid");
                grad_ys = _DefaultGradYs(grad_ys, ys, colocate_gradients_with_ops, gradient_uid);

                /** 
                 * The approach we take here is as follows: Create a list of all ops in the
                 * subgraph between the ys and xs.  Visit these ops in reverse order of ids
                 * to ensure that when we visit an op the gradients w.r.t its outputs have
                 * been collected.  Then aggregate these gradients if needed, call the op's
                 * gradient function, and add the generated gradients to the gradients for
                 * its input.
                 **/

                // Initialize the pending count for ops in the connected subgraph from ys
                // to the xs.
                var to_ops = ys.Select(x => x.op).ToList();
                var from_ops = xs.Select(x => x.op).ToList();
                var stop_gradient_ops = stop_gradients.Select(x => x.op).ToList();
                (var reachable_to_ops, var pending_count, var loop_state) = _PendingCount(to_ops, from_ops, colocate_gradients_with_ops, new List<object>(), xs);
                
                foreach(var (y, grad_y) in Python.zip(ys, grad_ys))
                    _SetGrad(grads, y, grad_y);

                // Initialize queue with to_ops.
                var queue = new Queue<Operation>();
                // Add the ops in 'to_ops' into the queue.
                var to_ops_set = new List<Operation>();
                foreach (var op in to_ops)
                {
                    // 'ready' handles the case where one output gradient relies on
                    // another output's gradient.
                    bool ready = !pending_count.ContainsKey(op.Name) || pending_count[op.Name] == 0;
                    if(ready && !to_ops_set.Contains(op) && reachable_to_ops.Contains(op))
                    {
                        to_ops_set.Add(op);
                        queue.Enqueue(op);
                    }
                }

                var stop_ops = _StopOps(from_ops, stop_gradient_ops, pending_count, xs);
                while(queue.Count > 0)
                {
                    // generate gradient subgraph for op.
                    var op = queue.Dequeue();
                    _maybe_colocate_with(op, gradient_uid, colocate_gradients_with_ops);
                    //if (loop_state != null)
                    //loop_state.EnterGradWhileContext(op, before: true);
                    var out_grads = _AggregatedGrads(grads, op, gradient_uid, loop_state, aggregation_method);

                    Tensor[] in_grads = null;
                    var is_partitioned_call = _IsPartitionedCall(op);
                    var is_func_call = false;
                    var has_out_grads = true;
                    if (has_out_grads && !stop_ops.Contains(op))
                    {
                        if (is_func_call)
                        {

                        }
                        else
                        {
                            // A grad_fn must be defined, either as a function or as None
                            // for ops that do not have gradients.
                            var grad_fn = ops.get_gradient_function(op);

                            Python.with<ops.name_scope>(new ops.name_scope(op.Name + "_grad"), delegate
                            {
                                if (grad_fn != null)
                                {
                                    in_grads = _MaybeCompile(grad_scope, op, out_grads[0], null, grad_fn);
                                    _VerifyGeneratedGradients(in_grads, op);
                                }
                            });
                            
                        }
                    }

                    var inputs = (List<Tensor>)_NonEagerInputs(op, xs);
                    foreach (var (t_in, in_grad) in Python.zip(inputs, in_grads))
                    {
                        if(in_grad != null)
                        {
                            in_grad.shape = t_in.shape;
                            _SetGrad(grads, t_in, in_grad);
                        }
                    }

                    // Update pending count for the inputs of op and enqueue ready ops.
                    _UpdatePendingAndEnqueueReady(grads, op, queue, pending_count, loop_state, xs);
                }
            });

            return xs.Select(x => _GetGrad(grads, x)).ToArray();
        }

        private static void _UpdatePendingAndEnqueueReady(Dictionary<string, Tensor[][]> grads, 
            Operation op, 
            Queue<Operation> queue, 
            Dictionary<string ,int> pending_count,
            object loop_state,
            Tensor[] xs)
        {

        }

        private static void _VerifyGeneratedGradients(Tensor[] grads, Operation op)
        {
            if (grads.Count() != op.inputs._inputs.Count())
                throw new ValueError($"Num gradients {grads.Length} generated for op {op.node_def} do not match num " +
                    $"inputs {op.inputs._inputs.Count()}");
        }

        private static Tensor[] _MaybeCompile(string scope, Operation op, Tensor out_grads, Action func, Func<Operation, Tensor, (Tensor, Tensor)> grad_fn)
        {
            var in_grads = grad_fn(op, out_grads);
            return new Tensor[] { in_grads.Item1, in_grads.Item2 };
        }

        private static bool _IsPartitionedCall(Operation op)
        {
            return op.OpType == "PartitionedCall" || op.OpType == "StatefulPartitionedCall";
        }

        private static Tensor[] _AggregatedGrads(Dictionary<string, Tensor[][]> grads, Operation op, string gradient_uid, object loop_state, int aggregation_method = 0)
        {
            var out_grads = _GetGrads(grads, op);
            for(int i = 0; i < out_grads.Length; i++)
            {
                var out_grad = out_grads[i];
                if(loop_state != null)
                {

                }

                // Grads have to be Tensors or IndexedSlices

                // Aggregate multiple gradients, and convert [] to None.
                if(out_grad != null)
                {
                    if(out_grad.Length < 2)
                    {
                        string used = "nop";
                        return new Tensor[] { out_grad[0] };
                    }
                }
            }

            return null;
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

            foreach(var op in from_ops)
            {
                bool is_stop_op = true;
                foreach(var inp in _NonEagerInputs(op, xs))
                {
                    if(pending_count.ContainsKey(op.Name) && pending_count[op.Name] > 0)
                    {
                        is_stop_op = false;
                        break;
                    }
                }
                if (is_stop_op)
                    stop_ops.Add(op);
            }
            stop_ops.AddRange(stop_gradient_ops.Where(x => !stop_ops.Contains(x)));
            return stop_ops.ToArray();
        }

        private static Tensor _GetGrad(Dictionary<string, Tensor[][]> grads, Tensor t)
        {
            var op = t.op;
            if (!grads.ContainsKey(op.Name))
                return null;
            Tensor[][] op_grads = grads[op.Name];
            var t_grad = op_grads[t.value_index];
            return t_grad[0];
        }

        private static Tensor[][] _GetGrads(Dictionary<string, Tensor[][]> grads, Operation op)
        {
            if (grads.ContainsKey(op.Name))
                return grads[op.Name];
            else
                return op.outputs.Select(x => new Tensor[0]).ToArray();
        }

        /// <summary>
        /// Sets gradient "grad" in "grads" for tensor "t".
        /// </summary>
        /// <param name="grads"></param>
        /// <param name="t"></param>
        /// <param name="grad"></param>
        private static void _SetGrad(Dictionary<string, Tensor[][]> grads, Tensor t, Tensor grad)
        {
            var op = t.op;
            Tensor[][] op_grads = null;
            if (!grads.ContainsKey(op.Name))
            {
                op_grads = op.outputs.Select(x => new Tensor[1]).ToArray();
                grads[op.Name] = op_grads;
            }
            var t_grads = op_grads[t.value_index];
            // t_grads[0] = grad;
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

            for(int i = 0; i < grad_ys.Length; i++)
            {
                var grad_y = grad_ys[i];
                var y = ys[i];

                _maybe_colocate_with(y.op, gradient_uid, colocate_gradients_with_ops);

                if(grad_y == null)
                {
                    if (y.dtype.is_complex())
                        throw new TypeAccessException($"Gradients of complex tensors must set grad_ys (y.dtype = {y.dtype})");
                    var shape = array_ops.shape(y);
                    var constant = constant_op.constant(1.0, name: $"grad_ys_{i}");
                    var fill = gen_array_ops.fill(shape, constant);
                    new_grad_ys.Add(fill);
                }
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
        private static (Operation[], Dictionary<string, int>, object) _PendingCount(List<Operation> to_ops, List<Operation> from_ops, bool colocate_gradients_with_ops, List<object> func_graphs, Tensor[] xs)
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
            while(queue.Count > 0)
            {
                var op = queue.Dequeue();
                if (reached_ops.Contains(op))
                {
                    between_ops.Add(op);
                    between_op_list.Insert(between_op_list.Count, op);
                    // Clear the boolean so we won't add the inputs again.
                    reached_ops.Remove(op);
                    foreach (var inp in _NonEagerInputs(op, xs))
                        queue.Enqueue((inp as Tensor).op);
                }
            }
            // X in between_ops iff X is on a path of zero or more backpropagatable tensors
            // between from_ops and to_ops

            // 'loop_state' is None if there are no while loops.
            var loop_state = control_flow_ops.MaybeCreateControlFlowState(between_op_list, between_ops, colocate_gradients_with_ops);

            var pending_count = new Dictionary<string, int>();
            foreach (var op in between_op_list)
            {
                foreach(Tensor x in _NonEagerInputs(op, xs))
                {
                    if (between_ops.Contains(x.op))
                        if (pending_count.ContainsKey(x.op.Name))
                            pending_count[x.op.Name] += 1;
                        else
                            pending_count[x.op.Name] = 1;
                }
            }

            return (reachable_to_ops.ToArray(), pending_count, loop_state);
        }

        private static InputList _NonEagerInputs(Operation op, Tensor[] xs)
        {
            return op.inputs;
        }

        /// <summary>
        /// Mark all ops reached from "from_ops"
        /// </summary>
        /// <param name="from_ops"></param>
        /// <param name="reached_ops"></param>
        /// <param name="func_graphs"></param>
        private static void _MarkReachedOps(List<Operation> from_ops, List<Operation> reached_ops, List<object> func_graphs)
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
                        var c = _Consumers(output, func_graphs).ToList();
                        c.ForEach(x => queue.Enqueue(x));
                    }
                }
            }
        }

        /// <summary>
        /// Returns the consumers of t, crossing closure boundaries where necessary.
        /// </summary>
        /// <param name="t"></param>
        /// <param name="func_graphs"></param>
        private static List<Operation> _Consumers(Tensor t, List<object> func_graphs)
        {
            var consumers = t.consumers();
            return consumers;
        }

        private static List<Tensor> _AsList(object ys)
        {
            List<Tensor> ret = null;

            switch (ys)
            {
                case Tensor value:
                    ret = new List<Tensor> { value };
                    break;
                case List<Tensor> value:
                    ret = value;
                    break;
            }

            return ret;
        }
    }
}
