using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Handler for structured fetches.
    /// </summary>
    public class _FetchHandler
    {
        private _ElementFetchMapper _fetch_mapper;
        private List<object> _fetches = new List<object>();
        private List<bool> _ops = new List<bool>();
        private List<object> _final_fetches = new List<object>();

        public _FetchHandler(Graph graph, Tensor fetches, object feeds = null, object feed_handles = null)
        {
            _fetch_mapper = new _FetchMapper().for_fetch(fetches);
            foreach(var fetch in _fetch_mapper.unique_fetches())
            {
                switch (fetch)
                {
                    case Tensor val:
                        _assert_fetchable(graph, val.op);
                        _fetches.Add(fetch);
                        _ops.Add(false);
                        break;
                }

            }

            _final_fetches = _fetches;
        }

        private void _assert_fetchable(Graph graph, Operation op)
        {
            if (!graph.is_fetchable(op))
            {
                throw new Exception($"Operation {op.name} has been marked as not fetchable.");
            }
        }
    }
}
