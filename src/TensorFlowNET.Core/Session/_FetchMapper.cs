using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class _FetchMapper
    {
        public _ElementFetchMapper for_fetch(Tensor fetch)
        {
            var fetches = new List<Tensor> { fetch };

            return new _ElementFetchMapper(fetches, null);
        }
    }
}
