using System.Collections.Generic;
using static Tensorflow.Binding;

namespace Tensorflow.Train
{
    public class TrainingUtil
    {
        public static IVariableV1 create_global_step(Graph graph = null)
        {
            graph = graph ?? ops.get_default_graph();
            if (get_global_step(graph) != null)
                throw new ValueError("global_step already exists.");

            // Create in proper graph and base name_scope.
            var g = graph.as_default();
            g.name_scope(null);
            var v = tf.compat.v1.get_variable(tf.GraphKeys.GLOBAL_STEP, new int[0], dtype: dtypes.int64,
                initializer: tf.zeros_initializer,
                trainable: false,
                aggregation: VariableAggregation.OnlyFirstReplica,
                collections: new List<string> { tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP });
            return v;
        }

        public static RefVariable get_global_step(Graph graph = null)
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

        public static Tensor _get_or_create_global_step_read(Graph graph = null)
        {
            graph = graph ?? ops.get_default_graph();
            var global_step_read_tensor = _get_global_step_read(graph);
            if (global_step_read_tensor != null)
                return global_step_read_tensor;

            var global_step_tensor = get_global_step(graph);

            if (global_step_tensor == null)
                return null;

            var g = graph.as_default();
            g.name_scope(null);
            g.name_scope(global_step_tensor.Op.name + "/");
            // using initialized_value to ensure that global_step is initialized before
            // this run. This is needed for example Estimator makes all model_fn build
            // under global_step_read_tensor dependency.
            var global_step_value = global_step_tensor.initialized_value();
            ops.add_to_collection(tf.GraphKeys.GLOBAL_STEP_READ_KEY, global_step_value + 0);

            return _get_global_step_read(graph);
        }

        private static Tensor _get_global_step_read(Graph graph = null)
        {
            graph = graph ?? ops.get_default_graph();
            var global_step_read_tensors = graph.get_collection<Tensor>(tf.GraphKeys.GLOBAL_STEP_READ_KEY);
            if (global_step_read_tensors.Count > 1)
                throw new RuntimeError($"There are multiple items in collection {tf.GraphKeys.GLOBAL_STEP_READ_KEY}. " +
                    "There should be only one.");

            if (global_step_read_tensors.Count == 1)
                return global_step_read_tensors[0];

            return null;
        }
    }
}
