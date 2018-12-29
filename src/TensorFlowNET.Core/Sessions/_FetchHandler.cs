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
        private List<Tensor> _fetches = new List<Tensor>();
        private List<bool> _ops = new List<bool>();
        private List<Tensor> _final_fetches = new List<Tensor>();
        private List<object> _targets = new List<object>();

        public _FetchHandler(Graph graph, Tensor fetches, Dictionary<Tensor, object> feeds = null, object feed_handles = null)
        {
            _fetch_mapper = new _FetchMapper().for_fetch(fetches);
            foreach(var fetch in _fetch_mapper.unique_fetches())
            {
                switch (fetch)
                {
                    case Tensor val:
                        _assert_fetchable(graph, val.op);
                        _fetches.Add(val);
                        _ops.Add(false);
                        break;
                }

            }

            _final_fetches = _fetches;
        }

        public object build_results(Session session, object[] results)
        {
            return _fetch_mapper.build_results(results);
        }

        private void _assert_fetchable(Graph graph, Operation op)
        {
            if (!graph.is_fetchable(op))
            {
                throw new Exception($"Operation {op.name} has been marked as not fetchable.");
            }
        }

        public List<Tensor> fetches()
        {
            return _final_fetches;
        }

        public List<Object> targets()
        {
            return _targets;
        }
    }
}
