﻿/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using NumSharp;
using System.Collections.Generic;

namespace Tensorflow
{
    public class _FetchMapper
    {
        protected List<ITensorOrOperation> _unique_fetches = new List<ITensorOrOperation>();
        protected List<int[]> _value_indices = new List<int[]>();
        public static _FetchMapper for_fetch(object fetch)
        {
            var fetches = fetch.GetType().IsArray ? (object[])fetch : new object[] { fetch };

            if(fetch is List<string> fetches1)
                return new _ListFetchMapper(fetches1.ToArray());
            if (fetch.GetType().IsArray)
                return new _ListFetchMapper(fetches);
            else
                return new _ElementFetchMapper(fetches, (List<NDArray> fetched_vals) => fetched_vals[0]);
        }

        public virtual NDArray build_results(List<NDArray> values)
        {
            var type = values[0].GetType();
            var nd = new NDArray(type, values.Count);
            nd.ReplaceData(values.ToArray());
            return nd;
        }

        public virtual List<ITensorOrOperation> unique_fetches()
        {
            return _unique_fetches;
        }
    }
}
