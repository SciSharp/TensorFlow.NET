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

using System.Collections.Generic;
using Tensorflow.IO;

namespace Tensorflow
{
    public static partial class tf
    {
        public static GFile gfile = new GFile();
        public static Tensor read_file(string filename, string name = null) => gen_io_ops.read_file(filename, name);

        public static gen_image_ops image => new gen_image_ops();

        public static void import_graph_def(GraphDef graph_def,
            Dictionary<string, Tensor> input_map = null,
            string[] return_elements = null,
            string name = null,
            OpList producer_op_list = null) => importer.import_graph_def(graph_def, input_map, return_elements, name, producer_op_list);
    }
}
