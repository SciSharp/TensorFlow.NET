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
using System;
using System.Collections.Generic;
using NumSharp.Backends;

namespace Tensorflow
{
    /// <summary>
    /// Handler for structured fetches.
    /// </summary>
    public class _FetchHandler
    {
        private _FetchMapper _fetch_mapper;
        private List<Tensor> _fetches = new List<Tensor>();
        private List<bool> _ops = new List<bool>();
        private List<Tensor> _final_fetches = new List<Tensor>();
        private List<object> _targets = new List<object>();

        public _FetchHandler(Graph graph, object fetches, Dictionary<object, object> feeds = null, Action feed_handles = null)
        {
            _fetch_mapper = _FetchMapper.for_fetch(fetches, graph: graph);
            foreach(var fetch in _fetch_mapper.unique_fetches())
            {
                switch (fetch)
                {
                    case Operation val:
                        _assert_fetchable(graph, val);
                        _targets.Add(val);
                        _ops.Add(true);
                        break;
                    case Tensor val:
                        _assert_fetchable(graph, val.op);
                        _fetches.Add(val);
                        _ops.Add(false);
                        break;
                    default:
                        throw new NotImplementedException("_FetchHandler fetch");
                }

            }

            _final_fetches = _fetches;
        }

        public NDArray[] build_results(BaseSession session, NDArray[] tensor_values)
        {
            var full_values = new List<NDArray>();
            if (_final_fetches.Count != tensor_values.Length)
                throw new InvalidOperationException("_final_fetches mismatch tensor_values");

            int i = 0;
            int j = 0;
            foreach(var is_op in _ops)
            {
                if (is_op)
                {
                    if(tensor_values.Length > 0)
                    {
                        switch (tensor_values[0].typecode)
                        {
                            case NPTypeCode.Int32:
                                full_values.Add(float.NaN);
                                break;
                            case NPTypeCode.Single:
                                full_values.Add(float.NaN);
                                break;
                            case NPTypeCode.Double:
                                full_values.Add(float.NaN);
                                break;
                            case NPTypeCode.String:
                                full_values.Add(float.NaN);
                                break;
                            case NPTypeCode.Char:
                                full_values.Add(float.NaN);
                                break;
                            case NPTypeCode.Byte:
                                full_values.Add(float.NaN);
                                break;
                            default:
                                throw new NotImplementedException($"build_results tensor_values[0] {tensor_values[0].dtype.Name}");
                        }
                    }
                    else
                    {
                        full_values.Add(null);
                    }
                }
                else
                {
                    var value = tensor_values[j];
                    j += 1;
                    if (value.ndim == 0)
                    {
                        switch (value.typecode)
                        {
                            case NPTypeCode.Int16:
                                full_values.Add(value.GetValue<short>(0));
                                break;
                            case NPTypeCode.Int32:
                                full_values.Add(value.GetValue<int>(0));
                                break;
                            case NPTypeCode.Int64:
                                full_values.Add(value.GetValue<long>(0));
                                break;
                            case NPTypeCode.Single:
                                full_values.Add(value.GetValue<float>(0));
                                break;
                            case NPTypeCode.Double:
                                full_values.Add(value.GetValue<double>(0));
                                break;
                            /*case "String":
                                full_values.Add(value.Data<byte>()[0]);
                                break;*/
                            default:
                                throw new NotImplementedException($"build_results tensor_values[0] {tensor_values[0].dtype.Name}");
                        }
                    }
                    else
                    {
                        full_values.Add(value[np.arange(0, value.shape[0])]);
                    }
                }
                i += 1;
            }

            if (j != tensor_values.Length)
                throw new InvalidOperationException("j mismatch tensor_values");

            return _fetch_mapper.build_results(full_values);
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

        public List<object> targets()
        {
            return _targets;
        }
    }
}
