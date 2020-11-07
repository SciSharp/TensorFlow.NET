using System;
using System.Linq;
using static Tensorflow.Binding;

namespace Tensorflow.Graphs
{
    public class AutoGraph
    {
        public Func<Tensor, Tensor> to_graph(Func<Tensor, Tensor> func)
        {
            string func_name = $"autograph_{Guid.NewGuid()}_{func.Method.Name}";

            // IntPtr func_handle;
            using (var graph = new FuncGraph(func_name))
            {
                var input = tf.placeholder(tf.int32);
                var output = func(input);

                var opers = graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
                var func_handle = graph.ToGraph(opers,
                    new Operation[] { input },
                    new Operation[] { output },
                    null);
            }

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
            string func_name = $"autograph_{Guid.NewGuid()}_{func.Method.Name}";

            // IntPtr func_handle;
            using (var graph = new FuncGraph(func_name))
            {
                var input1 = tf.placeholder(tf.int32);
                var input2 = tf.placeholder(tf.int32);
                var output = func(input1, input2);

                var opers = graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
                var func_handle = graph.ToGraph(opers,
                    new Operation[] { input1, input2 },
                    new Operation[] { output },
                    null);
            }

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
