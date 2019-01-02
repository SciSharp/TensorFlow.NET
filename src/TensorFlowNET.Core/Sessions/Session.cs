using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class Session : BaseSession
    {
        private IntPtr _handle;

        public Session(string target = "", Graph graph = null)
        {
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
    }
}
