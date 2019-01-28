using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class Session : BaseSession, IPython
    {
        private IntPtr _handle;
        public Status Status { get; }
        public SessionOptions Options { get; }

        public Session(string target = "", Graph graph = null)
        {
            Status = new Status();
            if(graph == null)
            {
                graph = tf.get_default_graph();
            }
            Options = new SessionOptions();
            _handle = c_api.TF_NewSession(graph, Options, Status);
            Status.Check();
        }

        public Session(IntPtr handle)
        {
            _handle = handle;
        }

        public Session(Graph graph, SessionOptions opts, Status s)
        {
            _handle = c_api.TF_NewSession(graph, opts, s);
        }

        public static implicit operator IntPtr(Session session) => session._handle;
        public static implicit operator Session(IntPtr handle) => new Session(handle);

        public void Dispose()
        {
            Options.Dispose();
            Status.Dispose();
            c_api.TF_DeleteSession(_handle, Status);
        }

        public void __enter__()
        {
            
        }

        public void __exit__()
        {
            
        }
    }
}
