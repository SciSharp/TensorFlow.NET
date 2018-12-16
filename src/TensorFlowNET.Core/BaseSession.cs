using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Core
{
    public class BaseSession
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

        public virtual byte[] run(Tensor fetches)
        {
            return _run(fetches);
        }

        private unsafe byte[] _run(Tensor fetches)
        {
            var status = new Status();

            c_api.TF_SessionRun(_session,
                run_options: null,
                inputs: new TF_Input[] { },
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
