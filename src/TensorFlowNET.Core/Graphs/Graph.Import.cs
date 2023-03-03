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

using System.IO;
using System.Runtime.InteropServices;

namespace Tensorflow
{
    public partial class Graph
    {
        public unsafe TF_Output[] ImportGraphDefWithReturnOutputs(Buffer graph_def, ImportGraphDefOptions opts, Status s)
        {
            as_default();
            var num_return_outputs = opts.NumReturnOutputs;
            var return_outputs = new TF_Output[num_return_outputs];
            int size = Marshal.SizeOf<TF_Output>();
            var return_output_handle = Marshal.AllocHGlobal(size * num_return_outputs);

            c_api.TF_GraphImportGraphDefWithReturnOutputs(_handle, graph_def, opts, return_output_handle, num_return_outputs, s);

            var tf_output_ptr = (TF_Output*)return_output_handle;
            for (int i = 0; i < num_return_outputs; i++)
                return_outputs[i] = *(tf_output_ptr + i);

            Marshal.FreeHGlobal(return_output_handle);

            return return_outputs;
        }

        public bool Import(string file_path, string prefix = "")
        {
            var bytes = File.ReadAllBytes(file_path);
            return Import(bytes, prefix: prefix);
        }

        public bool Import(byte[] bytes, string prefix = "")
        {
            var opts = new ImportGraphDefOptions();
            var status = new Status();
            var graph_def = new Buffer(bytes);

            c_api.TF_ImportGraphDefOptionsSetPrefix(opts, prefix);
            c_api.TF_GraphImportGraphDef(_handle, graph_def, opts, status);
            status.Check(true);
            return status.Code == TF_Code.TF_OK;
        }

        public Graph ImportGraphDef(string file_path, string name = null)
        {
            as_default();
            var graph_def = GraphDef.Parser.ParseFrom(File.ReadAllBytes(file_path));
            importer.import_graph_def(graph_def, name: name);
            return this;
        }
    }
}
