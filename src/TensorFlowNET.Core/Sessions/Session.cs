using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class Session : BaseSession, IPython
    {
        private IntPtr _handle;
        public Status Status  = new Status();
        public SessionOptions Options { get; }
        public Graph graph;

        public Session(string target = "", Graph graph = null)
        {
            if(graph == null)
            {
                graph = tf.get_default_graph();
            }
            this.graph = graph;
            Options = new SessionOptions();
            _handle = c_api.TF_NewSession(graph, Options, Status);
            Status.Check();
        }

        public Session(IntPtr handle)
        {
            _handle = handle;
        }

        public Session(Graph g, SessionOptions opts = null, Status s = null)
        {
            if (s == null)
                s = Status;
            graph = g;
            Options = opts == null ? new SessionOptions() : opts;
            _handle = c_api.TF_NewSession(graph, Options, s);
            Status.Check(true);
        }

        public Session as_default()
        {
            tf.defaultSession = this;
            return this;
        }

        public static Session LoadFromSavedModel(string path)
        {
            var graph = c_api.TF_NewGraph();
            var status = new Status();
            var opt = c_api.TF_NewSessionOptions();

            var buffer = new TF_Buffer();
            var sess = c_api.TF_LoadSessionFromSavedModel(opt, IntPtr.Zero, path, new string[0], 0, graph, ref buffer, status);

            //var bytes = new Buffer(buffer.data).Data;
            //var meta_graph = MetaGraphDef.Parser.ParseFrom(bytes);

            status.Check();

            new Graph(graph).as_default();

            return sess;
        }

        public static implicit operator IntPtr(Session session) => session._handle;
        public static implicit operator Session(IntPtr handle) => new Session(handle);

        public void close()
        {
            Dispose();
        }

        public void Dispose()
        {
            Options.Dispose();
            c_api.TF_DeleteSession(_handle, Status);
            Status.Dispose();
        }

        public void __enter__()
        {
            
        }

        public void __exit__()
        {
            
        }
    }
}
