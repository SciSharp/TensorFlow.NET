using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public class _ListFetchMapper : _FetchMapper
    {
        private _FetchMapper[] _mappers;

        public _ListFetchMapper(object[] fetches)
        {
            _mappers = fetches.Select(fetch => _FetchMapper.for_fetch(fetch)).ToArray();
            _unique_fetches.AddRange(fetches);
        }
    }
}
