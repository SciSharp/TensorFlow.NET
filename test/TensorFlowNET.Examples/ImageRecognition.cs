using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Examples
{
    public class ImageRecognition : Python, IExample
    {
        public void Run()
        {
            var graph = new Graph();
            //import GraphDef from pb file
            graph.Import("tmp/tensorflow_inception_graph.pb");
            with<Session>(tf.Session(graph), sess =>
            {
                var labels = File.ReadAllLines("tmp/imagenet_comp_graph_label_strings.txt");
                var files = Directory.GetFiles("img");
                foreach(var file in files)
                {
                    var tensor = new Tensor(File.ReadAllBytes(file));
                }
            });
        }
    }
}
