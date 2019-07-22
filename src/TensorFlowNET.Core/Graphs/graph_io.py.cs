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
using System.IO;

namespace Tensorflow
{
    public class graph_io
    {
        public static string write_graph(Graph graph, string logdir, string name, bool as_text = true)
        {
            var graph_def = graph.as_graph_def();
            string path = Path.Combine(logdir, name);
            if (as_text)
                File.WriteAllText(path, graph_def.ToString());
            else
                File.WriteAllBytes(path, graph_def.ToByteArray());
            return path;
        }

        public static string write_graph(MetaGraphDef graph_def, string logdir, string name, bool as_text = true)
        {
            string path = Path.Combine(logdir, name);
            if (as_text)
                File.WriteAllText(path, graph_def.ToString());
            else
                File.WriteAllBytes(path, graph_def.ToByteArray());
            return path;
        }
    }
}
