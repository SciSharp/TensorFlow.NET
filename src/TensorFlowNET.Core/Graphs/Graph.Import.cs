﻿/*****************************************************************************
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

using System.IO;
using System.Runtime.InteropServices;

namespace Tensorflow
{
    public partial class Graph
    {
        public unsafe TF_Output[] ImportGraphDefWithReturnOutputs(Buffer graph_def, ImportGraphDefOptions opts, Status s)
        {
            var num_return_outputs = opts.NumReturnOutputs;
            var return_outputs = new TF_Output[num_return_outputs];
            int size = Marshal.SizeOf<TF_Output>();
            var return_output_handle = Marshal.AllocHGlobal(size * num_return_outputs);

            c_api.TF_GraphImportGraphDefWithReturnOutputs(_handle, graph_def, opts, return_output_handle, num_return_outputs, s);
            for (int i = 0; i < num_return_outputs; i++)
            {
                var handle = return_output_handle + i * size;
                return_outputs[i] = Marshal.PtrToStructure<TF_Output>(handle);
            }

            return return_outputs;
        }

        public Status Import(string file_path)
        {
            var bytes = File.ReadAllBytes(file_path);
            var graph_def = new Tensorflow.Buffer(bytes);
            var opts = c_api.TF_NewImportGraphDefOptions();
            c_api.TF_GraphImportGraphDef(_handle, graph_def, opts, Status);
            return Status;
        }

        public Status Import(byte[] bytes)
        {
            var graph_def = new Tensorflow.Buffer(bytes);
            var opts = c_api.TF_NewImportGraphDefOptions();
            c_api.TF_GraphImportGraphDef(_handle, graph_def, opts, Status);
            return Status;
        }

        public static Graph ImportFromPB(string file_path)
        {
            var graph = tf.Graph().as_default();
            var graph_def = GraphDef.Parser.ParseFrom(File.ReadAllBytes(file_path));
            importer.import_graph_def(graph_def);
            return graph;
        }
    }
}
