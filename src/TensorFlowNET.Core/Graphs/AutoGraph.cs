using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Graphs
{
    public class AutoGraph
    {
        public Func<Tensor, Tensor, Tensor> to_graph(Func<Tensor, Tensor, Tensor> func)
        {
            string func_name = $"autograph_{Guid.NewGuid()}_{func.Method.Name}";
            tf.compat.v1.disable_eager_execution();
            // IntPtr func_handle;
            using(var graph = new FuncGraph(func_name))
            {
                graph.as_default();
                var input1 = tf.placeholder(tf.int32);
                var input2 = tf.placeholder(tf.int32);
                var output = func(input1, input2);

                var opers = graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
                var func_handle = graph.ToGraph(opers,
                    new Operation[] { input1, input2 },
                    new Operation[] { output },
                    null);

                c_api.TFE_ContextAddFunction(tf.Context.Handle, func_handle, tf.Status.Handle);
            }

            tf.enable_eager_execution();

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
