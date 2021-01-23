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

using Google.Protobuf;
using NumSharp;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using Tensorflow.Util;
using static Tensorflow.Binding;

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

        public BaseSession(string target = "", Graph g = null, ConfigProto config = null, Status status = null)
        {
            _graph = g ?? ops.get_default_graph();
            if (!_graph.building_function)
                _graph.as_default();
            _target = Encoding.UTF8.GetBytes(target);

            using (var opts = new SessionOptions(target, config))
            {
                lock (Locks.ProcessWide)
                {
                    status = status ?? new Status();
                    _handle = c_api.TF_NewSession(_graph, opts.Handle, status.Handle);
                    status.Check(true);
                }
            }
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
            var results = _run(fetche, feed_dict);
            return fetche is Tensor ? results[0] : null;
        }

        public virtual (NDArray, NDArray, NDArray, NDArray, NDArray) run(
            (ITensorOrOperation, ITensorOrOperation, ITensorOrOperation, ITensorOrOperation, ITensorOrOperation) fetches,
            params FeedItem[] feed_dict)
        {
            var results = _run(new object[] { fetches.Item1, fetches.Item2, fetches.Item3, fetches.Item4, fetches.Item5 }, feed_dict);
            return (results[0], results[1], results[2], results[3], results[4]);
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
            var feed_items = feed_dict == null ? new FeedItem[0] : feed_dict.Keys.OfType<object>().Select(key => new FeedItem(key, feed_dict[key])).ToArray();
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
                    //var target_dtype = subfeed_t.dtype.as_numpy_typecode(); // subfeed_dtype was never used
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
            int i = 0;
            foreach (var x in feed_dict)
            {
                if (x.Key is Tensor key)
                {
                    switch (x.Value)
                    {
                        case Tensor v:
                            if (v.dtype != key.dtype)
                                throw new ValueError($"Tensor {v} does not match the expected dtype {key.dtype}, actual dtype: {v.dtype}");
                            feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), v);
                            break;
                        case NDArray v:
                            feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype));
                            break;
                        case IntPtr v:
                            var tensor = new Tensor(v);
                            if (tensor.dtype != key.dtype)
                                throw new ValueError($"Tensor {v} does not match the expected dtype {key.dtype}, actual dtype: {tensor.dtype}");

                            feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), tensor);
                            break;
#if _REGEN
                        // @formatter:off — disable formatter after this line
                        %types = ["bool", "sbyte", "byte", "short", "ushort", "int", "uint", "long", "ulong", "float", "double", "Complex"]
                        %foreach types%
                        case #1 v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case #1[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        %
                        // @formatter:on — enable formatter after this line
#else
                        // @formatter:off — disable formatter after this line
                        case bool v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case bool[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case sbyte v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case sbyte[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case byte v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case byte[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case short v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case short[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case ushort v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case ushort[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case int v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case int[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case uint v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case uint[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case long v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case long[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case ulong v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case ulong[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case float v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case float[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case double v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case double[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case Complex v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        case Complex[] v: feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype)); break;
                        // @formatter:on — enable formatter after this line
#endif

                        case string v:
                            feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), TensorConverter.ToTensor(v, key.dtype));
                            break;
                        default:
                            throw new NotImplementedException($"feed_dict data type {x.Value?.GetType().Name ?? "<null>"}");
                    }
                }
            }

            var fetches = fetch_list.Select(x => x._as_tf_output()).ToArray();
            //var targets = target_list;
            return _call_tf_sessionrun(feeds, fetches, target_list);
        }


        private unsafe NDArray[] _call_tf_sessionrun(KeyValuePair<TF_Output, Tensor>[] feed_dict, TF_Output[] fetch_list, List<Operation> target_list)
        {
            // Ensure any changes to the graph are reflected in the runtime.
            _extend_graph();

            var status = tf.Status;

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
                status: status.Handle);

            status.Check(true);

            var result = new NDArray[fetch_list.Length];

            for (int i = 0; i < fetch_list.Length; i++)
                result[i] = fetchValue(output_values[i]);

            return result;
        }

        private static unsafe NDArray fetchValue(IntPtr output)
        {
            var tensor = new Tensor(output);
            return tensor.numpy();
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
        { }

        protected override void DisposeUnmanagedResources(IntPtr handle)
        {
            lock (Locks.ProcessWide)
                using (var status = new Status())
                {
                    c_api.TF_DeleteSession(handle, status.Handle);
                    status.Check(true);
                }
        }
    }
}
