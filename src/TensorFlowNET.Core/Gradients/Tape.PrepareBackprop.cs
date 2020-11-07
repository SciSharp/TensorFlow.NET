using System.Collections.Generic;
using Tensorflow.Util;
using static Tensorflow.tensorflow;

namespace Tensorflow.Gradients
{
    public partial class Tape
    {
        public BackpropInitialState PrepareBackprop(long[] target,
            TensorTape tensor_tape,
            OpTape<BackwardFunction, TapeTensor> op_tape,
            UnorderedSet<long> sources_set,
            bool persistent_tape)
        {
            BackpropInitialState result = new BackpropInitialState();
            var tensor_stack = new Queue<long>(target);
            while (tensor_stack.Count > 0)
            {
                var tensor_id = tensor_stack.Dequeue();

                if (!tensor_tape.find(tensor_id, out var op_id))
                    continue;

                if (op_id == -1 ||
                    !op_tape.find(op_id, out var op_it) ||
                    result.op_tape.find(op_id, out var result_op_it))
                    continue;

                result.op_tape.emplace(op_id, op_it);

                foreach (var it in op_it.input_tensor_id)
                {
                    if (result.tensor_usage_counts.find(it))
                        result.tensor_usage_counts[it]++;
                    else
                    {
                        result.tensor_usage_counts[it] = 1;
                        if (tensor_tape.find(it))
                            tensor_stack.Enqueue(it);
                    }
                }

                if (!persistent_tape)
                    op_tape.Remove(op_id);
            }

            foreach (var pair in result.tensor_usage_counts)
            {
                if (tensor_tape.find(pair.Key, out var it) && it != -1)
                    result.op_missing_tensor[it] += 1;
            }

            if (!persistent_tape)
            {
                // Call destructors for all unneeded gradient functions and
                // clear the op_tape. We can clear the tape because ownership of
                // backward functions that will be used for gradient computation
                // has been transferred to `result`.
                /*for (const auto&op_pair : *op_tape) {
                    op_pair.second.backward_function_deleter(
                        op_pair.second.backward_function);
                }*/
                op_tape.Clear();
            }

            return result;
        }
    }
}
