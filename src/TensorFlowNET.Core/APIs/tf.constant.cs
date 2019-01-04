using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Tensor constant(NDArray nd, string name = "Const", bool verify_shape = false)
        {
            var t = constant_op.Create(nd, name, verify_shape);
            /*var graph = tf.get_default_graph();
            var tensor = new Tensor(nd);
            var op = graph.NewOperation("Const", name, tensor);*/
            
            return t;
        }
    }
}
