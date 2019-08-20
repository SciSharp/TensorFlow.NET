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
using Tensorflow.Train;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public train_internal train { get; } = new train_internal();

        public class train_internal
        {
            public Optimizer GradientDescentOptimizer(float learning_rate) 
                => new GradientDescentOptimizer(learning_rate);

            public Optimizer AdamOptimizer(float learning_rate, string name = "Adam") 
                => new AdamOptimizer(learning_rate, name: name);

            public Saver Saver(VariableV1[] var_list = null) => new Saver(var_list: var_list);

            public string write_graph(Graph graph, string logdir, string name, bool as_text = true) 
                => graph_io.write_graph(graph, logdir, name, as_text);

            public Saver import_meta_graph(string meta_graph_or_file,
                bool clear_devices = false,
                string import_scope = "") => saver._import_meta_graph_with_return_elements(meta_graph_or_file,
                    clear_devices,
                    import_scope).Item1;

            public (MetaGraphDef, Dictionary<string, RefVariable>) export_meta_graph(string filename = "",
                bool as_text = false,
                bool clear_devices = false,
                bool clear_extraneous_savers = false,
                bool strip_default_attrs = false) => meta_graph.export_scoped_meta_graph(filename: filename,
                    as_text: as_text,
                    clear_devices: clear_devices,
                    clear_extraneous_savers: clear_extraneous_savers,
                    strip_default_attrs: strip_default_attrs);
        }
    }
}
