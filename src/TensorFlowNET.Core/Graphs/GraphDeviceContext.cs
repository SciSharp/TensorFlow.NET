using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Graphs
{
    public class GraphDeviceContext : ITensorFlowObject
    {
        private Graph _graph;

        public GraphDeviceContext(Graph graph, string device_name)
        {
            _graph = graph;
        }

        public void __enter__()
        {

        }

        public void __exit__()
        {

        }

        public void Dispose()
        {

        }
    }
}
