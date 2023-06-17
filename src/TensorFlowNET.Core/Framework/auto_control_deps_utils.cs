using Tensorflow.Graphs;

namespace Tensorflow.Framework
{
    internal static class auto_control_deps_utils
    {
        public static readonly string READ_ONLY_RESOURCE_INPUTS_ATTR = "_read_only_resource_inputs";
        public static List<int> get_read_only_resource_input_indices_graph(FuncGraph func_graph)
        {
            List<int> result = new List<int>();
            // A cache to store the read only resource inputs of an Op.
            // Operation -> ObjectIdentitySet of resource handles.
            Dictionary<Operation, HashSet<Tensor>> opReadOnlyResourceInputs =
                new Dictionary<Operation, HashSet<Tensor>>();

            for (int inputIndex = 0; inputIndex < func_graph.Inputs.Length; inputIndex++)
            {
                Tensor t = func_graph.Inputs[inputIndex];
                if (t.dtype != dtypes.resource)
                    continue;

                bool readOnly = true;
                foreach (var op in t.consumers())
                {
                    if (opReadOnlyResourceInputs.ContainsKey(op))
                    {
                        if (!opReadOnlyResourceInputs[op].Contains(t))
                        {
                            readOnly = false;
                            break;
                        }
                    }
                    else
                    {
                        List<int> indices = _get_read_only_resource_input_indices_op(op);
                        opReadOnlyResourceInputs[op] = new HashSet<Tensor>(
                            indices.Select(i => op.inputs[i]));
                        if (!opReadOnlyResourceInputs[op].Contains(t))
                        {
                            readOnly = false;
                            break;
                        }
                    }
                }

                if (readOnly)
                    result.Add(inputIndex);
            }

            return result;
        }

        private static List<int> _get_read_only_resource_input_indices_op(Operation op)
        {
            // ignore the RESOURCE_READ_OPS

            int[] read_only_input_indices;

            try
            {
                read_only_input_indices = op.get_attr<int[]>(READ_ONLY_RESOURCE_INPUTS_ATTR);
            }
            catch (InvalidArgumentError)
            {
                return new List<int>();
            }

            int read_only_index = 0;
            List<int> result = new();
            for (int i = 0; i < op.inputs.Length; i++)
            {
                if (read_only_index >= read_only_input_indices.Length)
                {
                    break;
                }
                if (op.inputs[i].dtype != dtypes.resource)
                {
                    continue;
                }
                if (read_only_index < read_only_input_indices.Length && i == read_only_input_indices[read_only_index])
                {
                    result.Add(i);
                    read_only_index++;
                }
            }
            return result;
        }
    }
}
