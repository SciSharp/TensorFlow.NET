using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace Tensorflow.Gradients
{
    public partial class Tape
    {
        static readonly int kMinAggregateCount = 4;
        static readonly int kMinAggregateBytes = 128 * 1024 * 1024;
        private static UnorderedMap<string, UnorderedSet<int>> _functionsAcceptingNoneForIndicesMap;

        static Tape()
        {
            _functionsAcceptingNoneForIndicesMap = new();
            _functionsAcceptingNoneForIndicesMap.Add("SoftmaxCrossEntropyWithLogits", new UnorderedSet<int>(new[] { 1 }));
            _functionsAcceptingNoneForIndicesMap.Add("SparseSoftmaxCrossEntropyWithLogits", new UnorderedSet<int>(new[] { 1 }));
            _functionsAcceptingNoneForIndicesMap.Add("FusedBatchNorm", new UnorderedSet<int>(new[] { 1, 2, 3, 4 }));
        }

        public Tensor[] ComputeGradient(long[] target_tensor_ids,
            long[] source_tensor_ids,
            UnorderedMap<long, TapeTensor> sources_that_are_targets,
            List<Tensor> output_gradients, 
            bool build_default_zeros_grads)
        {
            UnorderedSet<long> sources_set = new(source_tensor_ids);
            BackpropInitialState state = PrepareBackprop(target_tensor_ids, tensor_tape_, op_tape_, sources_set, Persistent);
            var op_stack = InitialStack(state.op_tape, state.op_missing_tensor);
            var gradients = InitialGradients(target_tensor_ids, sources_that_are_targets, output_gradients, tensor_tape_, state.op_tape);
            UnorderedMap<long, long> gradients_size = new();
            while(op_stack.Count > 0)
            {
                long op = op_stack.Dequeue();
                if(!state.op_tape.TryGetValue(op, out var op_it))
                {
                    continue;
                }
                var trace = op_it;
                state.op_tape.erase(op);
                List<Tensor> out_gradients = new();
                List<long> unneeded_gradients = new();
                for(int i = 0, end = trace.input_tensor_id.Length; i < end; i++)
                {
                    long in_tensor_id = trace.input_tensor_id[i];
                    if(!tensor_tape_.find(in_tensor_id) && !sources_set.find(in_tensor_id))
                    {
                        unneeded_gradients.Add(i);
                    }
                }

                bool any_gradient_nonzero = false;
                List<int> zero_indices = new();
                for(int i = 0, end = trace.output_tensor_info.Length; i < end; i++)
                {
                    long id = trace.output_tensor_info[i].GetID();
                    if(!gradients.TryGetValue(id, out var grad_it))
                    {
                        out_gradients.Add(null);
                        if (build_default_zeros_grads)
                        {
                            if(!_functionsAcceptingNoneForIndicesMap.TryGetValue(trace.op_type, out var func_name_it) ||
                                !func_name_it.find(i))
                            {
                                zero_indices.Add(i);
                            }
                        }
                    }
                    else
                    {
                        any_gradient_nonzero = true;
                        Tensor new_gradients;
                        if (grad_it.Count == 1)
                        {
                            new_gradients = grad_it[0];
                        }
                        else
                        {
                            new_gradients = AggregateGradients(grad_it);
                        }
                        if (!sources_set.find(id))
                        {
                            gradients.Remove(id);
                        }
                        else
                        {
                            grad_it.Clear();
                            grad_it.Add(new_gradients);
                            // MarkAsResult
                        }
                        out_gradients.Add(new_gradients);
                    }
                }

                Tensor[] in_gradients = new Tensor[0];
                if (any_gradient_nonzero)
                {
                    foreach(var i in zero_indices)
                    {
                        out_gradients[i] = trace.output_tensor_info[i].ZerosLike();
                    }
                    in_gradients = CallBackwardFunction(trace.backward_function, unneeded_gradients, out_gradients);
                }
                else
                {
                    out_gradients.Clear();
                }
                
                for(int i = 0, end = in_gradients.Length; i < end; i++)
                {
                    long id = trace.input_tensor_id[i];
                    if (in_gradients[i] is not null)
                    {
                        var unaggregated_grads = gradients.SetDefault(id, new List<Tensor>());
                        unaggregated_grads.Add(in_gradients[i]);
                        if(unaggregated_grads.Count > kMinAggregateCount)
                        {
                            if(!gradients_size.TryGetValue(id, out var size))
                            {
                                size = NumElements(unaggregated_grads[0]);
                                gradients_size.emplace(id, size);
                            }
                            if(unaggregated_grads.Count * size * 4 > kMinAggregateBytes)
                            {
                                Tensor grad = AggregateGradients(unaggregated_grads);
                                unaggregated_grads.Clear();
                                unaggregated_grads.Add(grad);
                            }
                        }
                    }
                    if(!state.tensor_usage_counts.find(id))
                    {
                        continue;
                    }
                    state.tensor_usage_counts[id]--;
                    if(state.tensor_usage_counts[id] > 0)
                    {
                        continue;
                    }
                    if (!tensor_tape_.TryGetValue(id, out var tape_it))
                    {
                        if (gradients.find(id))
                        {
                            gradients.erase(id);
                        }
                        continue;
                    }
                    long op_id = tape_it;
                    if(op_id == -1)
                    {
                        continue;
                    }
                    if(state.op_missing_tensor.find(op_id))
                    {
                        state.op_missing_tensor[op_id]--;
                        if(state.op_missing_tensor[op_id] == 0)
                        {
                            op_stack.Enqueue(op_id);
                        }
                    }
                }
            }

            if(state.op_tape.Count > 0)
            {
                throw new RuntimeError("Invalid tape state.");
            }
            Tensor[] result = new Tensor[source_tensor_ids.Length];
            for(int i = 0; i < source_tensor_ids.Length; i++)
            {
                long tensor_id = source_tensor_ids[i];
                if(!gradients.TryGetValue(tensor_id, out var grad_it))
                {
                    result[i] = null;
                }
                else
                {
                    if(grad_it.Count > 1)
                    {
                        Tensor grad = AggregateGradients(grad_it);
                        grad_it.Clear();
                        grad_it.Add(grad);
                    }
                    result[i] = grad_it[0];
                }
            }
            return result;
        }

        UnorderedMap<string, UnorderedSet<int>> FunctionsAcceptingNoneForIndicesMap()
        {
            return _functionsAcceptingNoneForIndicesMap;
        }

        UnorderedMap<long, List<Tensor>> InitialGradients(long[] target_tensor_ids,
            UnorderedMap<long, TapeTensor> sources_that_are_targets,
            List<Tensor> output_gradients,
            TensorTape tensor_tape,
            OpTape op_tape)
        {
            var result = new UnorderedMap<long, List<Tensor>>();
            for(int i = 0, end = target_tensor_ids.Length; i < end; i++)
            {
                long id = target_tensor_ids[i];
                if( output_gradients is null ||output_gradients.Count == 0 || output_gradients[i] is null)
                {
                    if(tensor_tape.TryGetValue(id, out var tensor_it) && tensor_it != -1)
                    {
                        if(!op_tape.TryGetValue(tensor_it, out var op_it))
                        {
                            throw new RuntimeError("Internal state of the gradient tape is invalid: " +
                                "failed to find operation producing a tensor.");
                        }
                        bool found = false;
                        for(int j = 0; j < op_it.output_tensor_info.Length; j++)
                        {
                            if (op_it.output_tensor_info[j].GetID() == id)
                            {
                                found = true;
                                Tensor ones_like = BuildOnesLike(op_it.output_tensor_info[j]);
                                result.SetDefault(id, new List<Tensor>()).Add(ones_like);
                                break;
                            }
                        }
                        if (!found)
                        {
                            throw new RuntimeError("Internal state of the gradient tape is invalid: " +
                                "none of operations outputs match expected tensor.");
                        }
                    }
                    else
                    {
                        if(sources_that_are_targets.TryGetValue(id, out var source_tensor))
                        {
                            Tensor ones_like = BuildOnesLike(source_tensor);
                            result.SetDefault(id, new List<Tensor>()).Add(ones_like);
                        }
                    }
                }
                else
                {
                    result.SetDefault(id, new List<Tensor>()).Add(output_gradients[i]);
                }
            }

            return result;
        }

        Queue<long> InitialStack(OpTape op_tape,
            UnorderedMap<long, long> op_missing_tensor)
        {
            var result = new Queue<long>();
            foreach (var op_entry in op_tape)
            {
                if (!op_missing_tensor.find(op_entry.Key))
                    result.Enqueue(op_entry.Key);
            }
            return result;
        }

        Tensor BuildOnesLike(TapeTensor t)
        {
            return t.OnesLike();
        }

        Tensor AggregateGradients(List<Tensor> gradient_tensors)
        {
            if(gradient_tensors.Count == 0)
            {
                return gradient_tensors[0];
            }
            return tf.add_n(gradient_tensors.ToArray());
        }

        void DeleteGradient(Tensor gradient)
        {
            // Do not do anything here. Because GC will collect it when it has no reference.
        }

        long NumElements(Tensor tensor) => 1;
    }
}
