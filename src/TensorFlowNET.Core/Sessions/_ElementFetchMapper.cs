/*****************************************************************************
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

using Tensorflow.NumPy;
using System;
using System.Collections.Generic;

namespace Tensorflow
{
    /// <summary>
    /// Fetch mapper for singleton tensors and ops.
    /// </summary>
    public class _ElementFetchMapper : _FetchMapper
    {
        private Func<List<NDArray>, object> _contraction_fn;

        public _ElementFetchMapper(object[] fetches, Func<List<NDArray>, object> contraction_fn, Graph graph = null)
        {
            var g = graph ?? ops.get_default_graph();

            foreach (var fetch in fetches)
            {
                var el = g.as_graph_element(fetch, allow_tensor: true, allow_operation: true);
                _unique_fetches.Add(el);
            }

            _contraction_fn = contraction_fn;
        }

        /// <summary>
        /// Build results matching the original fetch shape.
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public override NDArray[] build_results(List<NDArray> values)
        {
            NDArray[] result = null;

            if (values.Count > 0)
            {
                var ret = _contraction_fn(values);
                switch (ret)
                {
                    case NDArray value:
                        result = new[] { value };
                        break;
                    case bool value:
                        result = new[] { NDArray.Scalar(value) };
                        break;
                    case byte value:
                        result = new[] { NDArray.Scalar(value) };
                        break;
                    case int value:
                        result = new[] { NDArray.Scalar(value) };
                        break;
                    case float value:
                        result = new[] { NDArray.Scalar(value) };
                        break;
                    default:
                        break;
                }
            }

            return result;
        }
    }
}
