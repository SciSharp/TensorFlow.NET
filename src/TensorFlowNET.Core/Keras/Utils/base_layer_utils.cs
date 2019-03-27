using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Utils
{
    public class base_layer_utils
    {
        public static RefVariable make_variable(string name,
            int[] shape,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            IInitializer initializer = null,
            bool trainable = false)
        {
            throw new NotImplementedException("");
        }

        /// <summary>
        /// Makes a layer name (or arbitrary string) unique within a TensorFlow graph.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public static string unique_layer_name(string name)
        {
            int number = get_default_graph_uid_map();
            return $"{name}_{number}";
        }

        public static int get_default_graph_uid_map()
        {
            var graph = ops.get_default_graph();
            return graph._next_id();
        }
    }
}
