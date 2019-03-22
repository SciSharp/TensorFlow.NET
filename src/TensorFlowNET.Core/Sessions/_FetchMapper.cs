using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class _FetchMapper
    {
        protected List<object> _unique_fetches = new List<object>();

        public static _FetchMapper for_fetch(object fetch)
        {
            var fetches = fetch.GetType().IsArray ? (object[])fetch : new object[] { fetch };

            if (fetch.GetType().IsArray)
                return new _ListFetchMapper(fetches);
            else
                return new _ElementFetchMapper(fetches, (List<object> fetched_vals) => fetched_vals[0]);
        }

        public virtual NDArray build_results(List<object> values)
        {
            return values.ToArray();
        }

        public virtual List<object> unique_fetches()
        {
            return _unique_fetches;
        }
    }
}
