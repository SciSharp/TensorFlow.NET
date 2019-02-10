using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class _FetchMapper
    {
        public _ElementFetchMapper for_fetch(object fetch)
        {
            var fetches = new object[] { fetch };

            return new _ElementFetchMapper(fetches, (List<object> fetched_vals) =>
            {
                return fetched_vals[0];
            });
        }
    }
}
