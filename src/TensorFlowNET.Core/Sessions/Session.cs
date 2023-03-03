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

namespace Tensorflow;

public class Session : BaseSession
{
    public Session(string target = "", Graph g = null) : base(target, g, null)
    { }

    public Session(SafeSessionHandle handle, Graph g = null) : base(handle, g)
    { }

    public Session(Graph g, ConfigProto config = null, Status s = null) : base("", g, config, s)
    { }

    public Session as_default()
    {
        return ops.set_default_session(this);
    }

    public static Session LoadFromSavedModel(string path)
    {
        var graph = new Graph();
        var status = new Status();
        using var opt = c_api.TF_NewSessionOptions();

        var tags = new string[] { "serve" };

        var sess = c_api.TF_LoadSessionFromSavedModel(opt,
                IntPtr.Zero,
                path,
                tags,
                tags.Length,
                graph,
                IntPtr.Zero,
                status);
        status.Check(true);

        // load graph bytes
        // var data = new byte[buffer.length];
        // Marshal.Copy(buffer.data, data, 0, (int)buffer.length);
        // var meta_graph = MetaGraphDef.Parser.ParseFrom(data);*/
        return new Session(sess, g: graph);
    }

    public static implicit operator SafeSessionHandle(Session session) => session._handle;
    public static implicit operator Session(SafeSessionHandle handle) => new Session(handle);
}
