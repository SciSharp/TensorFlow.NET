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

using NumSharp;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using Google.Protobuf;
using NumSharp.Backends;
using Tensorflow.Util;

namespace Tensorflow
{
    public class BaseSession : DisposableObject
    {
        protected Graph _graph;
        protected bool _opened;
        protected bool _closed;
        protected int _current_version;
        protected byte[] _target;
        public Graph graph => _graph;

        public BaseSession(string target = "", Graph g = null, SessionOptions opts = null)
        {
            _graph = g is null ? ops.get_default_graph() : g;
            _graph.as_default();
            _target = UTF8Encoding.UTF8.GetBytes(target);

            SessionOptions newOpts = opts ?? new SessionOptions();

            var status = new Status();

            _handle = c_api.TF_NewSession(_graph, opts ?? newOpts, status);

            // dispose opts only if not provided externally.
            if (opts == null)
                newOpts.Dispose();

            status.Check(true);
        }

        public virtual void run(Operation op, params FeedItem[] feed_dict)
        {
            _run(op, feed_dict);
        }

        public virtual NDArray run(Tensor fetche, params FeedItem[] feed_dict)
        {
            return _run(fetche, feed_dict)[0];
        }

        public virtual NDArray run(ITensorOrOperation fetche, params FeedItem[] feed_dict)
        {
            return _run(fetche, feed_dict)[0];
        }

        public virtual (NDArray, NDArray, NDArray, NDArray) run((ITensorOrOperation, ITensorOrOperation, ITensorOrOperation, ITensorOrOperation) fetches, params FeedItem[] feed_dict)
        {
            var results = _run(new object[] { fetches.Item1, fetches.Item2, fetches.Item3, fetches.Item4 }, feed_dict);
            return (results[0], results[1], results[2], results[3]);
        }

        public virtual (NDArray, NDArray, NDArray) run((ITensorOrOperation, ITensorOrOperation, ITensorOrOperation) fetches, params FeedItem[] feed_dict)
        {
            var results = _run(new object[] { fetches.Item1, fetches.Item2, fetches.Item3 }, feed_dict);
            return (results[0], results[1], results[2]);
        }

        public virtual (NDArray, NDArray) run((ITensorOrOperation, ITensorOrOperation) fetches, params FeedItem[] feed_dict)
        {
            var results = _run(new object[] { fetches.Item1, fetches.Item2 }, feed_dict);
            return (results[0], results[1]);
        }

        public virtual NDArray[] run(object fetches, params FeedItem[] feed_dict)
        {
            return _run(fetches, feed_dict);
        }

        public virtual NDArray[] run(object fetches, Hashtable feed_dict = null)
        {
            var feed_items = feed_dict == null ? new FeedItem[0] :
                feed_dict.Keys.OfType<object>().Select(key => new FeedItem(key, feed_dict[key])).ToArray();
            return _run(fetches, feed_items);
        }

        private NDArray[] _run(object fetches, FeedItem[] feed_dict = null)
        {
            var feed_dict_tensor = new Dictionary<object, object>();
            //var feed_map = new Dictionary<object, object>();

            // Validate and process feed_dict.
            if (feed_dict != null)
            {
                foreach (var subfeed in feed_dict)
                {
                    var subfeed_t = _graph.as_graph_element(subfeed.Key, allow_tensor: true, allow_operation: false);
                    //var subfeed_dtype = subfeed_t.dtype.as_numpy_datatype(); // subfeed_dtype was never used
                    feed_dict_tensor[subfeed_t] = subfeed.Value;
                    //feed_map[subfeed_t.name] = (subfeed_t, subfeed.Value);
                }
            }

            // Create a fetch handler to take care of the structure of fetches.
            var fetch_handler = new _FetchHandler(_graph, fetches, feed_dict_tensor);

            // Run request and get response.
            // We need to keep the returned movers alive for the following _do_run().
            // These movers are no longer needed when _do_run() completes, and
            // are deleted when `movers` goes out of scope when this _run() ends.
            var _ = _update_with_movers();
            var final_fetches = fetch_handler.fetches();
            var final_targets = fetch_handler.targets();

            // We only want to really perform the run if fetches or targets are provided,
            // or if the call is a partial run that specifies feeds.
            var results = _do_run(final_targets.Select(x => (Operation)x).ToList(), final_fetches, feed_dict_tensor);

            return fetch_handler.build_results(this, results);
        }

        /// <summary>
        /// Runs a step based on the given fetches and feeds.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="target_list">A list of operations to be run, but not fetched.</param>
        /// <param name="fetch_list"></param>
        /// <param name="feed_dict"></param>
        /// <returns>
        /// A list of numpy ndarrays, corresponding to the elements of
        /// `fetch_list`.  If the ith element of `fetch_list` contains the
        /// name of an operation, the first Tensor output of that operation
        /// will be returned for that element.
        /// </returns>
        private NDArray[] _do_run(List<Operation> target_list, List<Tensor> fetch_list, Dictionary<object, object> feed_dict)
        {

            var feeds = new KeyValuePair<TF_Output, Tensor>[feed_dict.Count];
            var ignoreDispose = new bool[feed_dict.Count];
            int i = 0;
            foreach (var x in feed_dict)
            {
                if (x.Key is Tensor tensor)
                {
                    switch (x.Value)
                    {
                        case Tensor v: ignoreDispose[i] = true; feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), v); break;
                        case NDArray v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v, tensor.dtype)); break;
#if _REGEN
                    %types = ["sbyte", "byte", "short", "ushort", "int", "uint", "long", "ulong", "float", "double", "Complex"]
                    %foreach types%
                        case #1 v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case #1[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                    %
#else
                        case sbyte v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case sbyte[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case byte v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case byte[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case short v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case short[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case ushort v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case ushort[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case int v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case int[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case uint v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case uint[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case long v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case long[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case ulong v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case ulong[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case float v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case float[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case double v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case double[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case Complex v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case Complex[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
#endif
                        case bool v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor((byte) (v ? 1 : 0), TF_DataType.TF_BOOL)); break;
                        case string v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        case IntPtr v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(v)); break;
                        default:
                            throw new NotImplementedException($"feed_dict data type {x.Value?.GetType().Name ?? "<null>"}");
                    }
                }
            }

            var fetches = fetch_list.Select(x => x._as_tf_output()).ToArray();
            //var targets = target_list;
            try
            {
                return _call_tf_sessionrun(feeds, fetches, target_list);
            } finally
            {
                for (var idx = 0; idx < feeds.Length; idx++)
                {
                    if (ignoreDispose[idx])
                        continue;
                    feeds[idx].Value.Dispose();
                }
            }
        }

        private unsafe NDArray[] _call_tf_sessionrun(KeyValuePair<TF_Output, Tensor>[] feed_dict, TF_Output[] fetch_list, List<Operation> target_list)
        {
            // Ensure any changes to the graph are reflected in the runtime.
            _extend_graph();

            var status = new Status();

            var output_values = fetch_list.Select(x => IntPtr.Zero).ToArray();

            c_api.TF_SessionRun(_handle,
                run_options: null,
                inputs: feed_dict.Select(f => f.Key).ToArray(),
                input_values: feed_dict.Select(f => (IntPtr)f.Value).ToArray(),
                ninputs: feed_dict.Length,
                outputs: fetch_list,
                output_values: output_values,
                noutputs: fetch_list.Length,
                target_opers: target_list.Select(f => (IntPtr)f).ToArray(),
                ntargets: target_list.Count,
                run_metadata: IntPtr.Zero,
                status: status);

            status.Check(true);

            var result = new NDArray[fetch_list.Length];

            for (int i = 0; i < fetch_list.Length; i++)
                result[i] = fetchValue(output_values[i]);

            return result;
        }

        private static unsafe NDArray fetchValue(IntPtr output)
        {
            NDArray ret;
            using (var tensor = new Tensor(output))
            {
                var ndims = tensor.shape;
                var srcAddress = c_api.TF_TensorData(output).ToInt64();

                if (ndims.Length == 0)
                {
                    switch (tensor.dtype)
                    {
                        case TF_DataType.TF_BOOL:
                            ret = NDArray.Scalar(*(bool*) srcAddress);
                            break;
                        case TF_DataType.TF_STRING:
                            using (var reader = new CodedInputStream(new IntPtr(srcAddress).Stream(8, (long)tensor.bytesize)))
                                ret = NDArray.FromString(reader.ReadString());
                            break;
                        case TF_DataType.TF_UINT8:
                            ret = NDArray.Scalar(*(byte*) srcAddress);
                            break;
                        case TF_DataType.TF_INT16:
                            ret = NDArray.Scalar(*(short*) srcAddress);
                            break;
                        case TF_DataType.TF_INT32:
                            ret = NDArray.Scalar(*(int*) srcAddress);
                            break;
                        case TF_DataType.TF_INT64:
                            ret = NDArray.Scalar(*(long*) srcAddress);
                            break;
                        case TF_DataType.TF_UINT16:
                            ret = NDArray.Scalar(*(ushort*) srcAddress);
                            break;
                        case TF_DataType.TF_UINT32:
                            ret = NDArray.Scalar(*(uint*) srcAddress);
                            break;
                        case TF_DataType.TF_UINT64:
                            ret = NDArray.Scalar(*(ulong*) srcAddress);
                            break;
                        case TF_DataType.TF_FLOAT:
                            ret = NDArray.Scalar(*(float*) srcAddress);
                            break;
                        case TF_DataType.TF_DOUBLE:
                            ret = NDArray.Scalar(*(double*) srcAddress);
                            break;
                        default:
                            throw new NotImplementedException("can't fetch output");
                    }
                } else
                {
                    //var size = (long) tensor.size;
                    //var itemsize = (long) tensor.itemsize;
                    var bytesize = (long) tensor.bytesize;
                    var src = (void*) srcAddress;

#if _REGEN
		            #region Compute
		            switch (tensor.dtype)
		            {
			            %foreach except(supported_dtypes, "Char"),except(supported_dtypes_lowercase, "char"),except(supported_dtypes_TF_DataType,"TF_STRING")%
			            case TF_DataType.#3:
			            {
                            ret = new NDArray(NPTypeCode.#1, ndims, false);
                            System.Buffer.MemoryCopy(src, #(#3=="TF_STRING"|"(byte*)ret.Unsafe.Address + 8"|"ret.Unsafe.Address"), bytesize, bytesize);
				            break;
			            }
			            %
                        case TF_DataType.TF_STRING: 
                        {
                            //TODO:! This is not the way to handle string[], it should be done with TF_DecodeString
                            using (var reader = new CodedInputStream(new IntPtr(srcAddress).Stream(8, (long)tensor.bytesize)))
                                ret = NDArray.FromString(reader.ReadString());
                            break;
                        }
			            default:
				            throw new NotSupportedException();
		            }
		            #endregion
#else

		            #region Compute
		            switch (tensor.dtype)
		            {
			            case TF_DataType.TF_BOOL:
			            {
                            ret = new NDArray(NPTypeCode.Boolean, ndims, false);
                            System.Buffer.MemoryCopy(src, ret.Unsafe.Address, bytesize, bytesize);
				            break;
			            }
			            case TF_DataType.TF_UINT8:
			            {
                            ret = new NDArray(NPTypeCode.Byte, ndims, false);
                            System.Buffer.MemoryCopy(src, ret.Unsafe.Address, bytesize, bytesize);
				            break;
			            }
			            case TF_DataType.TF_INT16:
			            {
                            ret = new NDArray(NPTypeCode.Int16, ndims, false);
                            System.Buffer.MemoryCopy(src, ret.Unsafe.Address, bytesize, bytesize);
				            break;
			            }
			            case TF_DataType.TF_UINT16:
			            {
                            ret = new NDArray(NPTypeCode.UInt16, ndims, false);
                            System.Buffer.MemoryCopy(src, ret.Unsafe.Address, bytesize, bytesize);
				            break;
			            }
			            case TF_DataType.TF_INT32:
			            {
                            ret = new NDArray(NPTypeCode.Int32, ndims, false);
                            System.Buffer.MemoryCopy(src, ret.Unsafe.Address, bytesize, bytesize);
				            break;
			            }
			            case TF_DataType.TF_UINT32:
			            {
                            ret = new NDArray(NPTypeCode.UInt32, ndims, false);
                            System.Buffer.MemoryCopy(src, ret.Unsafe.Address, bytesize, bytesize);
				            break;
			            }
			            case TF_DataType.TF_INT64:
			            {
                            ret = new NDArray(NPTypeCode.Int64, ndims, false);
                            System.Buffer.MemoryCopy(src, ret.Unsafe.Address, bytesize, bytesize);
				            break;
			            }
			            case TF_DataType.TF_UINT64:
			            {
                            ret = new NDArray(NPTypeCode.UInt64, ndims, false);
                            System.Buffer.MemoryCopy(src, ret.Unsafe.Address, bytesize, bytesize);
				            break;
			            }
			            case TF_DataType.TF_DOUBLE:
			            {
                            ret = new NDArray(NPTypeCode.Double, ndims, false);
                            System.Buffer.MemoryCopy(src, ret.Unsafe.Address, bytesize, bytesize);
				            break;
			            }
			            case TF_DataType.TF_FLOAT:
			            {
                            ret = new NDArray(NPTypeCode.Single, ndims, false);
                            System.Buffer.MemoryCopy(src, ret.Unsafe.Address, bytesize, bytesize);
				            break;
			            }
                        case TF_DataType.TF_STRING:
                        {
                            throw new NotImplementedException();
                            //TODO:! This is not the way to handle string[], it should be done with TF_DecodeString
                            using (var reader = new CodedInputStream(new IntPtr(srcAddress).Stream(8, (long)tensor.bytesize)))
                                ret = NDArray.FromString(reader.ReadString());
                            break;
                        }
			            default:
				            throw new NotSupportedException();
		            }
		            #endregion
#endif
                }
            }

            return ret;
        }

        /// <summary>
        /// If a tensor handle that is fed to a device incompatible placeholder, 
        /// we move the tensor to the right device, generate a new tensor handle, 
        /// and update feed_dict to use the new handle.
        /// </summary>
        private List<object> _update_with_movers()
        {
            return new List<object> { };
        }

        private void _extend_graph()
        {

        }

        public void close()
        {
            Dispose();
        }

        protected override void DisposeUnmanagedResources(IntPtr handle)
        {
            using (var status = new Status())
            {
                c_api.TF_DeleteSession(handle, status);
                status.Check(true);
            }
        }
    }
}