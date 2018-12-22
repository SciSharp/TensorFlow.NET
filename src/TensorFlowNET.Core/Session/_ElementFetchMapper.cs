using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Fetch mapper for singleton tensors and ops.
    /// </summary>
    public class _ElementFetchMapper : _FetchMapper
    {
        private List<Object> _unique_fetches = new List<object>();
        private Action _contraction_fn;

        public _ElementFetchMapper(List<Tensor> fetches, Action contraction_fn)
        {
            foreach(var tensor in fetches)
            {
                var fetch = ops.get_default_graph().as_graph_element(tensor, allow_tensor: true, allow_operation: true);
                _unique_fetches.Add(fetch);
            }
        }

        public object build_results(object[] values)
        {
            return values[0];
        }

        public List<Object> unique_fetches()
        {
            return _unique_fetches;
        }
    }
}
