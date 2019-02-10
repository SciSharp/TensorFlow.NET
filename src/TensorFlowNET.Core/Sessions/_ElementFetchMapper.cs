using NumSharp.Core;
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
        private List<object> _unique_fetches = new List<object>();
        private Func<List<object>, object> _contraction_fn;

        public _ElementFetchMapper(object[] fetches, Func<List<object>, object> contraction_fn)
        {
            var g = ops.get_default_graph();
            ITensorOrOperation el = null;

            foreach(var fetch in fetches)
            {
                switch(fetch)
                {
                    case Tensor tensor:
                        el = g.as_graph_element(tensor, allow_tensor: true, allow_operation: true);
                        break;
                    case Operation op:
                        el = g.as_graph_element(op, allow_tensor: true, allow_operation: true);
                        break;
                    case String str:
                        // Looks like a Tensor name and can be a Tensor.
                        el = g._nodes_by_name[str];
                        break;
                    default:
                        throw new NotImplementedException("_ElementFetchMapper");
                }
            }

            _unique_fetches.Add(el);
            _contraction_fn = contraction_fn;
        }

        /// <summary>
        /// Build results matching the original fetch shape.
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public NDArray build_results(List<object> values)
        {
            NDArray result = null;

            if (values.Count > 0)
            {
                var ret = _contraction_fn(values);
                switch (ret)
                {
                    case NDArray value:
                        result = value;
                        break;
                    default:
                        break;
                }
            }

            return result;
        }

        public List<object> unique_fetches()
        {
            return _unique_fetches;
        }
    }
}
