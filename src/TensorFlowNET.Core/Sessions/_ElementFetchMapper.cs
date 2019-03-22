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
        private Func<List<object>, object> _contraction_fn;

        public _ElementFetchMapper(object[] fetches, Func<List<object>, object> contraction_fn)
        {
            var g = ops.get_default_graph();
            ITensorOrOperation el = null;

            foreach(var fetch in fetches)
            {
                el = g.as_graph_element(fetch, allow_tensor: true, allow_operation: true);
            }

            _unique_fetches.Add(el);
            _contraction_fn = contraction_fn;
        }

        /// <summary>
        /// Build results matching the original fetch shape.
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public override NDArray build_results(List<object> values)
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
                    case float fVal:
                        result = fVal;
                        break;
                    default:
                        break;
                }
            }

            return result;
        }
    }
}
