using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Util;
using static Tensorflow.tensorflow;

namespace Tensorflow.Gradients
{
    public partial class Tape
    {
        int kMinAggregateCount = 4;
        int kMinAggregateBytes = 128 * 1024 * 1024;

        public Tensor[] ComputeGradient(long[] target_tensor_ids,
            long[] source_tensor_ids,
            UnorderedMap<long, TapeTensor> sources_that_are_targets,
            Tensor[] output_gradients)
        {
            var result = new List<Tensor>(source_tensor_ids.Length);
            var sources_set = new UnorderedSet<long>(source_tensor_ids);
            var gradients_size = new UnorderedMap<long, long>();

            var state = PrepareBackprop(
                target_tensor_ids, tensor_tape_, op_tape_, sources_set, persistent_);
            var op_stack = InitialStack(state.op_tape, state.op_missing_tensor);
            var gradients = InitialGradients(target_tensor_ids, sources_that_are_targets,
                output_gradients,
                tensor_tape_,
                state.op_tape);

            while (!op_stack.empty())
            {
                var op = op_stack.Dequeue();
                if (!state.op_tape.find(op, out var trace))
                    continue;

                // Console.WriteLine($"ComputeGradient: {state.op_tape[op].op_type}");
                state.op_tape.erase(op);

                var out_gradients = new List<Tensor>(trace.output_tensor_info.Length);
                var unneeded_gradients = new List<long>();
                for (int i = 0; i < trace.input_tensor_id.Length; i++)
                {
                    var in_tensor_id = trace.input_tensor_id[i];
                    if (!tensor_tape_.find(in_tensor_id) &&
                        !sources_set.find(in_tensor_id))
                        unneeded_gradients.Add(i);
                }

                bool any_gradient_nonzero = false;
                var zero_indices = new List<int>();
                for (int i = 0; i < trace.output_tensor_info.Length; ++i)
                {
                    var id = trace.output_tensor_info[i].GetID();
                    if (!gradients.find(id, out var grad_it))
                    {
                        if (FunctionsAcceptingNoneForIndicesMap().find(trace.op_type, out var func_name_it) &&
                            func_name_it.find(i))
                        {
                            out_gradients.Add(null);
                        }
                        else
                        {
                            out_gradients.Add(null);
                            zero_indices.Add(i);
                        }
                    }
                    else
                    {
                        any_gradient_nonzero = true;
                        var new_gradients = grad_it.Count == 1 ? 
                            grad_it[0] :
                            gen_math_ops.add_n(grad_it.ToArray()); // vspace.AggregateGradients

                        if (!sources_set.find(id))
                            gradients.Remove(id);
                        else
                        {
                            grad_it.Clear();
                            grad_it.Add(new_gradients);
                            // vspace.MarkAsResult(new_gradients);
                        }
                        out_gradients.Add(new_gradients);
                    }
                }

                Tensor[] in_gradients;
                if (any_gradient_nonzero)
                {
                    foreach (var i in zero_indices)
                        out_gradients[i] = trace.output_tensor_info[i].ZerosLike();

                    in_gradients = CallBackwardFunction(trace.backward_function, 
                        unneeded_gradients, 
                        out_gradients);

                    if (in_gradients.Count() != trace.input_tensor_id.Count())
                        throw new RuntimeError($"Recorded operation '{trace.op_type}' returned too few gradients. Expected {trace.input_tensor_id.Length} but received {in_gradients.Count()}");
                    if (!persistent_)
                    {
                        // trace.backward_function_deleter(trace.backward_function);
                    }
                }
                else
                {
                    in_gradients = new Tensor[trace.input_tensor_id.Length];
                }

                for (int i = 0; i < in_gradients.Length; ++i)
                {
                    var id = trace.input_tensor_id[i];
                    if (in_gradients[i] != null)
                    {
                        var unaggregated_grads = gradients[id];
                        unaggregated_grads.Add(in_gradients[i]);
                        if (unaggregated_grads.Count > kMinAggregateCount)
                        {
                            if (!gradients_size.find(id, out var size))
                            {
                                size = (long)unaggregated_grads[0].size;
                                gradients_size.emplace(id, size);
                            }

                            if (unaggregated_grads.Count * size * 4 > kMinAggregateBytes)
                            {
                                throw new NotImplementedException("");
                            }
                        }
                    }

                    if (!state.tensor_usage_counts.find(id))
                        continue;

                    state.tensor_usage_counts[id]--;
                    if (state.tensor_usage_counts[id] > 0)
                        continue;

                    if (!tensor_tape_.find(id, out var tape_it))
                    {
                        if (gradients.find(id, out var grad_it))
                        {
                            // foreach (var g in grad_it)
                                // DeleteGradient(g);
                            gradients.erase(id);
                        }
                        continue;
                    }

                    var op_id = tape_it;
                    if (op_id == -1)
                        continue;

                    if(state.op_missing_tensor.find(op_id, out var missing_it))
                    {
                        state.op_missing_tensor[op_id]--;
                        if (state.op_missing_tensor[op_id] == 0)
                            op_stack.Enqueue(op_id);
                    }
                }
            }

            if (state.op_tape.Count > 0)
                throw new RuntimeError("Invalid tape state.");

            var used_gradient_ids = new List<long>(source_tensor_ids.Length);
            foreach (var id in source_tensor_ids)
            {
                if (!gradients.find(id, out var grad_it))
                    result.Add(null);
                else
                {
                    if(grad_it.Count > 1)
                    {
                        var grad = gen_math_ops.add_n(grad_it.ToArray());
                        grad_it.Clear();
                        grad_it.Add(grad);
                    }
                    result.Add(grad_it[0]);
                    used_gradient_ids.Add(id);
                }
            }

            /*foreach(var grad_pair in gradients)
            {
                if(!used_gradient_ids.Contains(grad_pair.Key))
                {
                    foreach(var g in grad_pair.Value)
                    {
                        vspace.DeleteGradient(g);
                    }
                }
            }*/

            return result.ToArray();
        }

        UnorderedMap<string, UnorderedSet<int>> FunctionsAcceptingNoneForIndicesMap()
        {
            var m = new UnorderedMap<string, UnorderedSet<int>>();
            m.Add("SoftmaxCrossEntropyWithLogits", new UnorderedSet<int>(new[] { 1 }));
            m.Add("SparseSoftmaxCrossEntropyWithLogits", new UnorderedSet<int>(new[] { 1 }));
            m.Add("FusedBatchNorm", new UnorderedSet<int>(new[] { 1, 2, 3, 4 }));
            return m;
        }

        UnorderedMapEnumerable<long, List<Tensor>> InitialGradients(long[] target_tensor_ids,
            UnorderedMap<long, TapeTensor> sources_that_are_targets,
            Tensor[] output_gradients,
            TensorTape tensor_tape,
            OpTape<BackwardFunction, TapeTensor> op_tape)
        {
            var result = new UnorderedMapEnumerable<long, List<Tensor>>();
            for (int i = 0; i < target_tensor_ids.Length; ++i)
            {
                var id = target_tensor_ids[i];
                if (output_gradients.Length == 0 || output_gradients[i] == null)
                {
                    if (tensor_tape.find(id, out var tensor_id) && tensor_id != -1)
                    {
                        if (!op_tape.find(tensor_tape[id], out var op_it))
                            throw new RuntimeError("Internal state of the gradient tape is invalid: " +
                                "failed to find operation producing a tensor");
                        bool found = false;
                        for (int j = 0; j < op_it.output_tensor_info.Length; ++j)
                        {
                            if (op_it.output_tensor_info[j].GetID() == id)
                            {
                                found = true;
                                var ones = op_it.output_tensor_info[j].OnesLike();
                                result[id].Add(ones);
                                break;
                            }
                        }

                        if (!found)
                        {
                            throw new ValueError("Internal state of the gradient tape is invalid: " +
                                "none of operations outputs match expected tensor");
                        }
                    }
                    else
                    {
                        if (sources_that_are_targets.find(id, out var source_tensor))
                            result[id].Add(source_tensor.OnesLike());
                    }
                }
                else
                {
                    result[id].Add(output_gradients[i]);
                }
            }

            return result;
        }

        Queue<long> InitialStack(OpTape<BackwardFunction, TapeTensor> op_tape,
            UnorderedMap<long, long> op_missing_tensor)
        {
            var result = new Queue<long>();
            foreach(var op_entry in op_tape)
            {
                if (!op_missing_tensor.find(op_entry.Key))
                    result.Enqueue(op_entry.Key);
            }
            return result;
        }
    }
}
