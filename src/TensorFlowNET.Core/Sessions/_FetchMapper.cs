using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class _FetchMapper<T>
    {
        public _ElementFetchMapper<T> for_fetch(T fetch)
        {
            var fetches = new List<T> { fetch };

            return new _ElementFetchMapper<T>(fetches, (List<object> fetched_vals) =>
            {
                return fetched_vals[0];
            });
        }
    }
}
