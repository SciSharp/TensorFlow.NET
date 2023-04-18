using Tensorflow.Graphs;

namespace Tensorflow
{
    public partial class Graph
    {
        public void _colocate_with_for_gradient(Operation op, string gradient_uid, bool ignore_existing = false)
        {

        }

        internal GraphOverrideGradientContext _override_gradient_function(Dictionary<string, Func<Operation, object[], Tensor[]>> gradient_function_map)
        {
            return new GraphOverrideGradientContext(this, gradient_function_map);
        }
    }
}
