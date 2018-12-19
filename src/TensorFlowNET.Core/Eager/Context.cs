using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Eager
{
    public class Context
    {
        public static int GRAPH_MODE = 0;
        public static int EAGER_MODE = 1;

        public int default_execution_mode;


    }
}
