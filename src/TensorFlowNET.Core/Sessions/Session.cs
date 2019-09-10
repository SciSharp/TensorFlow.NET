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

using System;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class Session : BaseSession, IObjectLife
    {
        public Session(string target = "", Graph g = null) : base(target, g, null)
        { }

        public Session(IntPtr handle, Graph g = null) : base("", g, null)
        {
            _handle = handle;
        }

        public Session(Graph g, SessionOptions opts = null, Status s = null) : base("", g, opts, s)
        { }

        public Session as_default()
        {
            tf._defaultSessionFactory.Value = this;
            return this;
        }

        public static Session LoadFromSavedModel(string path)
        {
            lock (Locks.ProcessWide)
            {
                var graph = c_api.TF_NewGraph();
                var status = new Status();
                var opt = new SessionOptions();

                var tags = new string[] {"serve"};
                var buffer = new TF_Buffer();

                var sess = c_api.TF_LoadSessionFromSavedModel(opt,
                    IntPtr.Zero,
                    path,
                    tags,
                    tags.Length,
                    graph,
                    ref buffer,
                    status);

                // load graph bytes
                // var data = new byte[buffer.length];
                // Marshal.Copy(buffer.data, data, 0, (int)buffer.length);
                // var meta_graph = MetaGraphDef.Parser.ParseFrom(data);*/
                status.Check(true);

                return new Session(sess, g: new Graph(graph)).as_default();
            }
        }

        public static implicit operator IntPtr(Session session) => session._handle;
        public static implicit operator Session(IntPtr handle) => new Session(handle);

        public void __enter__()
        {

        }

        public void __exit__()
        {

        }

        public void __init__()
        {
            
        }

        public void __del__()
        {
            
        }
    }
}
