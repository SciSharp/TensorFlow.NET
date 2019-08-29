using NumSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace TensorFlowNET.Examples.ImageProcessing.YOLO
{
    class Utils
    {
        public static Dictionary<int, string> read_class_names(string file)
        {
            var classes = new Dictionary<int, string>();
            foreach (var line in File.ReadAllLines(file))
                classes[classes.Count] = line;
            return classes;
        }

        public static NDArray get_anchors(string file)
        {
            return np.array(File.ReadAllText(file).Split(',')
                .Select(x => float.Parse(x))
                .ToArray()).reshape(3, 3, 2);
        }
    }
}
