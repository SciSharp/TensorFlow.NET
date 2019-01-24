using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Fetch mapper for singleton tensors and ops.
    /// </summary>
    public class _ElementFetchMapper<T> : _FetchMapper<T>
    {
        private List<object> _unique_fetches = new List<object>();
        private Func<List<object>> _contraction_fn;

        public _ElementFetchMapper(List<T> fetches, Func<List<object>> contraction_fn)
        {
            foreach(var fetch in fetches)
            {
                var g = ops.get_default_graph();
                var el = g.as_graph_element(fetch, allow_tensor: true, allow_operation: true);
                _unique_fetches.Add(el);
            }

            _contraction_fn = contraction_fn;
        }

        /// <summary>
        /// Build results matching the original fetch shape.
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public NDArray build_results(List<object> values)
        {
            if (values.Count == 0)
                return null;
            else
                return _contraction_fn(values);
        }

        public List<object> unique_fetches()
        {
            return _unique_fetches;
        }
    }
}
