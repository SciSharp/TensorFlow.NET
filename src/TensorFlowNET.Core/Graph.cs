using System;
using System.Collections.Generic;
using System.Text;

namespace TensorFlowNET.Core
{
    /// <summary>
    /// TensorFlow uses a dataflow graph to represent your computation in terms of the dependencies between individual operations. 
    /// This leads to a low-level programming model in which you first define the dataflow graph, 
    /// then create a TensorFlow session to run parts of the graph across a set of local and remote devices.
    /// https://www.tensorflow.org/guide/graphs
    /// </summary>
    public class Graph
    {
        public IntPtr TFGraph { get; set; }

        public Operation create_op()
        {
            var op = new Operation(this);

            return op;
        }
    }
}
