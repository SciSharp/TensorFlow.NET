using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Train
{
    public class TrainingUtil
    {
        public static RefVariable create_global_step(Graph graph)
        {
            graph = graph ?? ops.get_default_graph();
            if (get_global_step(graph) != null)
                throw new ValueError("global_step already exists.");

            // Create in proper graph and base name_scope.
            var g = graph.as_default();
            g.name_scope(null);
            var v = tf.get_variable(tf.GraphKeys.GLOBAL_STEP, new TensorShape(), dtype: dtypes.int64,
                initializer: tf.zeros_initializer,
                trainable: false,
                aggregation: VariableAggregation.OnlyFirstReplica,
                collections: new List<string> { tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP });
            return v;
        }

        public static RefVariable get_global_step(Graph graph)
        {
            graph = graph ?? ops.get_default_graph();
            RefVariable global_step_tensor = null;
            var global_step_tensors = graph.get_collection<RefVariable>(tf.GraphKeys.GLOBAL_STEP);
            if (global_step_tensors.Count == 1)
            {
                global_step_tensor = global_step_tensors[0];
            }
            else
            {
                try
                {
                    global_step_tensor = graph.get_tensor_by_name("global_step:0");
                }
                catch (KeyError)
                {
                    return null;
                }
            }
                
            return global_step_tensor;
        }
    }
}
