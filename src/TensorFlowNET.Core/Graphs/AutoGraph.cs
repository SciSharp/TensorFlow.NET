using System;
using System.Linq;
using static Tensorflow.Binding;

namespace Tensorflow.Graphs
{
    public class AutoGraph
    {
        public Func<Tensor, Tensor> to_graph(Func<Tensor, Tensor> func)
        {
            string func_name = $"{func.Method.Name}_{Guid.NewGuid()}";

            var graph = new FuncGraph(func_name);
            graph.as_default();

            var input = tf.placeholder(tf.int32);
            var output = func(input);

            var opers = graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
            graph.ToGraph(opers,
                new[] { input },
                new[] { output },
                null);
            graph.Exit();
            

            return (Tensor input) =>
            {
                var result = tf.Runner.TFE_Execute(tf.Context,
                    tf.Context.DeviceName,
                    func_name,
                    new[] { input },
                    null,
                    1);
                return result[0];
            };
        }

        public Func<Tensor, Tensor, Tensor> to_graph(Func<Tensor, Tensor, Tensor> func)
        {
            string func_name = $"{func.Method.Name}_{Guid.NewGuid()}";

            var graph = new FuncGraph(func_name);
            graph.as_default();

            var input1 = tf.placeholder(tf.int32);
            var input2 = tf.placeholder(tf.int32);
            var output = func(input1, input2);

            var opers = graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
            graph.ToGraph(opers,
                new[] { input1, input2 },
                new[] { output },
                null);
            graph.Exit();
            
            return (Tensor a, Tensor b) =>
            {
                var result = tf.Runner.TFE_Execute(tf.Context,
                    tf.Context.DeviceName,
                    func_name,
                    new[] { a, b },
                    null,
                    1);
                return result[0];
            };
        }
    }
}
