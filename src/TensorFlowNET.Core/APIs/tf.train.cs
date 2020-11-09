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
            public IVariableV1 create_global_step(Graph graph)
                => TrainingUtil.create_global_step(graph);

            public IVariableV1 get_global_step(Graph graph)
                => TrainingUtil.get_global_step(graph);

            public Optimizer GradientDescentOptimizer(float learning_rate)
                => new GradientDescentOptimizer(learning_rate);

            public Optimizer GradientDescentOptimizer(Tensor learning_rate)
                => new GradientDescentOptimizer(learning_rate);

            public Optimizer AdamOptimizer(float learning_rate, float epsilon = 1e-8f, string name = "Adam")
                => new AdamOptimizer(learning_rate, epsilon: epsilon, name: name);

            public Optimizer AdamOptimizer(float learning_rate, TF_DataType dtype, string name = "Adam")
                => new AdamOptimizer(learning_rate, name: name, dtype: dtype);

            public Optimizer AdamOptimizer(IVariableV1 learning_rate, string name = "Adam")
                => new AdamOptimizer(learning_rate.AsTensor(), name: name);

            public Optimizer AdamOptimizer(Tensor learning_rate, string name = "Adam")
                => new AdamOptimizer(learning_rate, name: name);

            public ExponentialMovingAverage ExponentialMovingAverage(float decay)
                => new ExponentialMovingAverage(decay);

            public Saver Saver(IVariableV1[] var_list = null, int max_to_keep = 5)
                => new Saver(var_list: var_list, max_to_keep: max_to_keep);

            public string write_graph(Graph graph, string logdir, string name, bool as_text = true)
                => graph_io.write_graph(graph, logdir, name, as_text);

            public Graph load_graph(string freeze_graph_pb)
                => saver.load_graph(freeze_graph_pb);

            public string freeze_graph(string checkpoint_dir, string output_pb_name, string[] output_node_names)
                => saver.freeze_graph(checkpoint_dir, output_pb_name, output_node_names);

            public Saver import_meta_graph(string meta_graph_or_file,
                bool clear_devices = false,
                string import_scope = "") => saver._import_meta_graph_with_return_elements(meta_graph_or_file,
                    clear_devices,
                    import_scope).Item1;

            public (MetaGraphDef, Dictionary<string, IVariableV1>) export_meta_graph(string filename = "",
                bool as_text = false,
                bool clear_devices = false,
                bool clear_extraneous_savers = false,
                bool strip_default_attrs = false) => meta_graph.export_scoped_meta_graph(filename: filename,
                    as_text: as_text,
                    clear_devices: clear_devices,
                    clear_extraneous_savers: clear_extraneous_savers,
                    strip_default_attrs: strip_default_attrs);

            public string latest_checkpoint(string checkpoint_dir, string latest_filename = null)
                => checkpoint_management.latest_checkpoint(checkpoint_dir, latest_filename: latest_filename);

            public CheckpointState get_checkpoint_state(string checkpoint_dir, string latest_filename = null)
                => checkpoint_management.get_checkpoint_state(checkpoint_dir, latest_filename: latest_filename);

            /*public Tensor polynomial_decay(float learning_rate,
                RefVariable global_step,
                float decay_steps,
                float end_learning_rate = 0.0001f,
                float power = 1.0f,
                bool cycle = false,
                string name = null)
            {
                var decayed = new PolynomialDecay(learning_rate,
                    decay_steps,
                    end_learning_rate: end_learning_rate,
                    power: power,
                    cycle: cycle,
                    name: name);

                var decayed_lr = decayed.__call__(global_step);

                return decayed_lr;
            }*/
        }
    }
}
