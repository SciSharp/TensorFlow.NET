using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras
{
    public class backend
    {
        public static void track_variable(RefVariable v)
        {

        }

        public static Graph get_graph()
        {
            return ops.get_default_graph();
        }
    }
}
