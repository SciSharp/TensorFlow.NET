using System.Collections.Generic;
using Tensorflow.Util;

namespace Tensorflow.Gradients
{
    public partial class Tape
    {
        public BackpropInitialState PrepareBackprop(long[] target,
            TensorTape tensor_tape,
            OpTape op_tape,
            UnorderedSet<long> sources_set,
            bool persistent_tape)
        {
            Stack<long> tensor_stack = new Stack<long>();
            foreach(var t in target)
            {
                tensor_stack.Push(t);
            }
            BackpropInitialState result = new BackpropInitialState();
            while(tensor_stack.Count > 0)
            {
                long tensor_id = tensor_stack.Pop();
                if(!tensor_tape.TryGetValue(tensor_id, out var op_id))
                {
                    continue;
                }
                if(op_id == -1 || !op_tape.TryGetValue(op_id, out var op_it) 
                    || result.op_tape.find(op_id))
                {
                    continue;
                }
                result.op_tape.emplace(op_id, op_it);
                foreach(var it in op_it.input_tensor_id)
                {
                    if(result.tensor_usage_counts.find(it))
                    {
                        result.tensor_usage_counts[it]++;
                    }
                    else
                    {
                        result.tensor_usage_counts[it] = 1;
                        if (tensor_tape.find(it))
                        {
                            tensor_stack.Push(it);
                        }
                    }
                }
                if (!persistent_tape)
                {
                    op_tape.erase(op_id);
                }
            }
            foreach(var pair in result.tensor_usage_counts)
            {
                if(tensor_tape.TryGetValue(pair.Key, out var it) && it != -1)
                {
                    result.op_missing_tensor[it]++;
                }
            }
            if (!persistent_tape)
            {
                op_tape.Clear();
            }
            return result;
        }
    }
}
