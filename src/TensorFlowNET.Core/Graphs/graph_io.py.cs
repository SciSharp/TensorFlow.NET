using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Tensorflow
{
    public class graph_io
    {
        public static string write_graph(Graph graph, string logdir, string name, bool as_text = true)
        {
            var def = graph._as_graph_def();
            string path = Path.Combine(logdir, name);
            string text = def.ToString();
            if (as_text)
                File.WriteAllText(path, text);

            return path;
        }
    }
}
