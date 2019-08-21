using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace TensorFlowNET.Examples.ImageProcessing.YOLO
{
    public class Config
    {
        string _root;
        public string CLASSES;

        public Config(string root)
        {
            _root = root;
            CLASSES = Path.Combine(_root, "data", "classes", "coco.names");
        }
    }
}
