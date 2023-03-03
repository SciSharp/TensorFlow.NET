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
using Tensorflow.Util;

namespace Tensorflow
{
    public partial class Graph
    {
        public Buffer ToGraphDef(Status s)
        {
            var buffer = new Buffer();
            c_api.TF_GraphToGraphDef(_handle, buffer, s);
            s.Check(true);

            return buffer;
        }

        private GraphDef _as_graph_def(bool add_shapes = false)
        {
            GraphDef def;
            var status = new Status();
            var buffer = ToGraphDef(status);
            status.Check(true);
            // limit size to 250M, recursion to max 100
            var inputStream = CodedInputStream.CreateWithLimits(buffer.DangerousMemoryBlock, 250 * 1024 * 1024, 100);
            def = GraphDef.Parser.ParseFrom(inputStream);

            // Strip the experimental library field iff it's empty.
            // if(def.Library.Function.Count == 0)

            return def;
        }

        public GraphDef as_graph_def(bool add_shapes = false)
            => _as_graph_def(add_shapes);
    }
}