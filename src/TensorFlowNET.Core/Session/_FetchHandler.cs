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

        public _FetchHandler(Graph graph, Tensor fetches, object feeds = null, object feed_handles = null)
        {
            _fetch_mapper = new _FetchMapper().for_fetch(fetches);
        }
    }
}
