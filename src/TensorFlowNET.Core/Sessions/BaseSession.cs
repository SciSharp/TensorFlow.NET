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
        private Graph _graph;
        private bool _opened;
        private bool _closed;
        private int _current_version;
        private byte[] _target;
        private IntPtr _session;

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
            _session = c_api.TF_NewSession(_graph.Handle, opts, status.Handle);

            c_api.TF_DeleteSessionOptions(opts);
        }

        public void Dispose()
        {
            
        }

        public virtual object run(Tensor fetches, Dictionary<Tensor, object> feed_dict = null)
        {
            var result = _run(fetches, feed_dict);

            return result;
        }

        private unsafe object _run(Tensor fetches, Dictionary<Tensor, object> feed_dict = null)
        {
            var feed_dict_tensor = new Dictionary<Tensor, NDArray>();

            if (feed_dict != null)
            {
                NDArray np_val = null;
                foreach (var feed in feed_dict)
                {
                    switch (feed.Value)
                    {
                        case float value:
                            np_val = np.asarray(value);
                            break;
                    }

                    feed_dict_tensor[feed.Key] = np_val;
                }
            }

            // Create a fetch handler to take care of the structure of fetches.
            var fetch_handler = new _FetchHandler(_graph, fetches);

            // Run request and get response.
            // We need to keep the returned movers alive for the following _do_run().
            // These movers are no longer needed when _do_run() completes, and
            // are deleted when `movers` goes out of scope when this _run() ends.
            var _ = _update_with_movers();
            var final_fetches = fetch_handler.fetches();
            var final_targets = fetch_handler.targets();

            // We only want to really perform the run if fetches or targets are provided,
            // or if the call is a partial run that specifies feeds.
            var results = _do_run(final_fetches);

            return fetch_handler.build_results(null, results);
        }

        private object[] _do_run(List<object> fetch_list)
        {
            var fetches = fetch_list.Select(x => (x as Tensor)._as_tf_output()).ToArray();

            return _call_tf_sessionrun(fetches);
        }

        private unsafe object[] _call_tf_sessionrun(TF_Output[] fetch_list)
        {
            // Ensure any changes to the graph are reflected in the runtime.
            _extend_graph();

            var status = new Status();

            var output_values = fetch_list.Select(x => IntPtr.Zero).ToArray();

            c_api.TF_SessionRun(_session,
                run_options: IntPtr.Zero,
                inputs: new TF_Output[] { },
                input_values: new IntPtr[] { },
                ninputs: 0,
                outputs: fetch_list,
                output_values: output_values,
                noutputs: fetch_list.Length,
                target_opers: new IntPtr[] { },
                ntargets: 0,
                run_metadata: IntPtr.Zero,
                status: status.Handle);

            var result = output_values.Select(x => c_api.TF_TensorData(x))
                .Select(x => (object)*(float*)x)
                .ToArray();

            return result;
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
