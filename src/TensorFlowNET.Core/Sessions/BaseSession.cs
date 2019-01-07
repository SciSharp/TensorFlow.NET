using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public class BaseSession : IDisposable
    {
        protected Graph _graph;
        protected bool _opened;
        protected bool _closed;
        protected int _current_version;
        protected byte[] _target;
        protected IntPtr _session;

        public BaseSession(string target = "", Graph graph = null)
        {
            if(graph is null)
            {
                _graph = ops.get_default_graph();
            }
            else
            {
                _graph = graph;
            }

            _target = UTF8Encoding.UTF8.GetBytes(target);
            var opts = c_api.TF_NewSessionOptions();
            var status = new Status();
            _session = c_api.TF_NewSession(_graph, opts, status);

            c_api.TF_DeleteSessionOptions(opts);
        }

        public void Dispose()
        {
            
        }

        public virtual object run(Tensor fetches, Dictionary<Tensor, NDArray> feed_dict = null)
        {
            var result = _run(fetches, feed_dict);

            return result;
        }

        private unsafe object _run(Tensor fetches, Dictionary<Tensor, NDArray> feed_dict = null)
        {
            var feed_dict_tensor = new Dictionary<Tensor, NDArray>();

            if (feed_dict != null)
            {
                foreach (var feed in feed_dict)
                {
                    feed_dict_tensor[feed.Key] = feed.Value;
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
            var results = _do_run(final_fetches, feed_dict_tensor);

            return fetch_handler.build_results(null, results);
        }

        private object[] _do_run(List<Tensor> fetch_list, Dictionary<Tensor, NDArray> feed_dict)
        {
            var feeds = feed_dict.Select(x => new KeyValuePair<TF_Output, Tensor>(x.Key._as_tf_output(), new Tensor(x.Value))).ToArray();
            var fetches = fetch_list.Select(x => x._as_tf_output()).ToArray();

            return _call_tf_sessionrun(feeds, fetches);
        }

        private unsafe object[] _call_tf_sessionrun(KeyValuePair<TF_Output, Tensor>[] feed_dict, TF_Output[] fetch_list)
        {
            // Ensure any changes to the graph are reflected in the runtime.
            _extend_graph();

            var status = new Status();

            var output_values = fetch_list.Select(x => IntPtr.Zero).ToArray();

            c_api.TF_SessionRun(_session,
                run_options: null,
                inputs: feed_dict.Select(f => f.Key).ToArray(),
                input_values: feed_dict.Select(f => (IntPtr)f.Value).ToArray(),
                ninputs: feed_dict.Length,
                outputs: fetch_list,
                output_values: output_values,
                noutputs: fetch_list.Length,
                target_opers: IntPtr.Zero,
                ntargets: 0,
                run_metadata: IntPtr.Zero,
                status: status);

            status.Check(true);

            object[] result = new object[fetch_list.Length];

            for (int i = 0; i < fetch_list.Length; i++)
            {
                var tensor = new Tensor(output_values[i]);
                Type type = tensor.dtype.as_numpy_datatype();
                var ndims = tensor.shape.Select(x => (int)x).ToArray();

                switch (tensor.dtype)
                {
                    case TF_DataType.TF_STRING:
                        {
                            // wired, don't know why we have to start from offset 9.
                            var bytes = tensor.Data();
                            var output = UTF8Encoding.Default.GetString(bytes, 9, bytes.Length - 9);
                            result[i] = fetchValue(tensor, ndims, output);
                        }
                        break;
                    case TF_DataType.TF_FLOAT:
                        {
                            var output = *(float*)c_api.TF_TensorData(output_values[i]);
                            result[i] = fetchValue(tensor, ndims, output);
                        }
                        break;
                    case TF_DataType.TF_INT16:
                        {
                            var output = *(short*)c_api.TF_TensorData(output_values[i]);
                            result[i] = fetchValue(tensor, ndims, output);
                        }
                        break;
                    case TF_DataType.TF_INT32:
                        {
                            var output = *(int*)c_api.TF_TensorData(output_values[i]);
                            result[i] = fetchValue(tensor, ndims, output);
                        }
                        break;
                    default:
                        throw new NotImplementedException("can't get output");
                }
            }

            return result;
        }

        private object fetchValue<T>(Tensor tensor, int[] ndims, T output)
        {
            if (tensor.NDims == 0)
            {
                return output;
            }
            else
            {
                return np.array(output).reshape(ndims);
            }
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
    }
}
