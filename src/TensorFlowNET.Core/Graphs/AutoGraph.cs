using System;
using System.Diagnostics;
using System.Linq;
using static Tensorflow.Binding;

namespace Tensorflow.Graphs
{
    public class AutoGraph
    {
        public Func<Tensor, Tensor> to_graph(Func<Tensor, Tensor> func, TF_DataType dtype = TF_DataType.TF_INT32)
        {
            string func_name = $"{func.Method.Name}_{ops.uid_function()}";

            var graph = new FuncGraph(func_name);
            graph.as_default();

            var input = tf.placeholder(dtype);
            var output = func(input);

            var opers = graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
            graph.ToGraph(opers,
                new[] { input },
                new[] { output },
                null);
            graph.Exit();
            

            return (Tensor input) =>
            {
                if (tf.executing_eagerly())
                {
                    var result = tf.Runner.TFE_Execute(tf.Context,
                        tf.Context.DeviceName,
                        func_name,
                        new[] { input },
                        null,
                        1);
                    return result[0];
                }
                var s = tf.Session(input.graph);
                var output = func(input);
                return output;
            };
        }

        public Func<Tensor, Tensor, Tensor> to_graph(Func<Tensor, Tensor, Tensor> func, params TF_DataType[] dtypes)
        {
            string func_name = $"{func.Method.Name}_{ops.uid_function()}";

            var graph = new FuncGraph(func_name);
            graph.as_default();

            var input1 = tf.placeholder(dtypes.Length >= 1 ? dtypes[0] : tf.int32);
            var input2 = tf.placeholder(dtypes.Length >= 2 ? dtypes[1] : tf.int32);
            var output = func(input1, input2);

            var opers = graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
            graph.ToGraph(opers,
                new[] { input1, input2 },
                new[] { output },
                null);
            graph.Exit();
            
            return (Tensor a, Tensor b) =>
            {
                if (tf.executing_eagerly())
                {
                    var result = tf.Runner.TFE_Execute(tf.Context,
                    tf.Context.DeviceName,
                    func_name,
                    new[] { a, b },
                    null,
                    1);
                    return result[0];
                }
                var s = tf.Session(a.graph);
                Debug.Assert(a.graph == b.graph);
                var output = func(a, b);
                return output;
            };
        }
    }
}
