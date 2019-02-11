using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static class train
        {
            public static Optimizer GradientDescentOptimizer(double learning_rate) => new GradientDescentOptimizer(learning_rate);

            public static Saver Saver() => new Saver();

            public static string write_graph(Graph graph, string logdir, string name, bool as_text = true) => graph_io.write_graph(graph, logdir, name, as_text);
        }
    }
}
