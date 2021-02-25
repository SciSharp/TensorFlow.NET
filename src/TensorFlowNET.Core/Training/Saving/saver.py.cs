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
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class saver
    {
        public static (Saver, object) _import_meta_graph_with_return_elements(string meta_graph_or_file,
            bool clear_devices = false,
            string import_scope = "",
            string[] return_elements = null)
        {
            var meta_graph_def = meta_graph.read_meta_graph_file(meta_graph_or_file);

            var (imported_vars, imported_return_elements) = meta_graph.import_scoped_meta_graph_with_return_elements(
                        meta_graph_def,
                        clear_devices: clear_devices,
                        import_scope: import_scope,
                        return_elements: return_elements);

            var saver = _create_saver_from_imported_meta_graph(
                meta_graph_def, import_scope, imported_vars);

            return (saver, imported_return_elements);
        }

        /// <summary>
        /// Return a saver for restoring variable values to an imported MetaGraph.
        /// </summary>
        /// <param name="meta_graph_def"></param>
        /// <param name="import_scope"></param>
        /// <param name="imported_vars"></param>
        /// <returns></returns>
        public static Saver _create_saver_from_imported_meta_graph(MetaGraphDef meta_graph_def,
            string import_scope,
            Dictionary<string, IVariableV1> imported_vars)
        {
            if (meta_graph_def.SaverDef != null)
            {
                // Infer the scope that is prepended by `import_scoped_meta_graph`.
                string scope = import_scope;
                var var_names = imported_vars.Keys.ToArray();
                if (var_names.Length > 0)
                {
                    var sample_key = var_names[0];
                    var sample_var = imported_vars[sample_key];
                    scope = string.Join("", sample_var.Name.Skip(sample_key.Length));
                }
                return new Saver(saver_def: meta_graph_def.SaverDef, name: scope);
            }
            else
            {
                if (variables._all_saveable_objects(scope: import_scope).Length > 0)
                {
                    // Return the default saver instance for all graph variables.
                    return new Saver();
                }
                else
                {
                    // If no graph variables exist, then a Saver cannot be constructed.
                    Binding.tf_output_redirect.WriteLine("Saver not created because there are no variables in the" +
                        " graph to restore");
                    return null;
                }
            }
        }

        public static string freeze_graph(string checkpoint_dir,
            string output_pb_name,
            string[] output_node_names)
        {
            var checkpoint = checkpoint_management.latest_checkpoint(checkpoint_dir);
            if (!File.Exists($"{checkpoint}.meta")) return null;

            string output_pb = Path.GetFullPath(Path.Combine(checkpoint_dir, "../", $"{output_pb_name}.pb"));

            using (var graph = tf.Graph())
            using (var sess = tf.Session(graph))
            {
                var saver = tf.train.import_meta_graph($"{checkpoint}.meta", clear_devices: true);
                saver.restore(sess, checkpoint);
                var output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                    graph.as_graph_def(),
                    output_node_names);
                Binding.tf_output_redirect.WriteLine($"Froze {output_graph_def.Node.Count} nodes.");
                File.WriteAllBytes(output_pb, output_graph_def.ToByteArray());
                return output_pb;
            }
        }

        public static Graph load_graph(string freeze_graph_pb, string name = "")
        {
            var bytes = File.ReadAllBytes(freeze_graph_pb);
            var graph = tf.Graph().as_default();
            importer.import_graph_def(GraphDef.Parser.ParseFrom(bytes),
                name: name);
            return graph;
        }
    }
}
