using NumSharp.Core;
using System;
using System.Collections.Generic;
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

        public virtual byte[] run(Tensor fetches, Dictionary<Tensor, object> feed_dict = null)
        {
            return _run(fetches, feed_dict);
        }

        private unsafe byte[] _run(Tensor fetches, Dictionary<Tensor, object> feed_dict = null)
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
            
            var status = new Status();

            c_api.TF_SessionRun(_session,
                run_options: null,
                inputs: new TF_Output[] { },
                input_values: new IntPtr[] { },
                ninputs: 1,
                outputs: new TF_Output[] { },
                output_values: new IntPtr[] { },
                noutputs: 1,
                target_opers: new IntPtr[] { },
                ntargets: 1,
                run_metadata: null,
                status: status.Handle);

            return null;
        }
    }
}
