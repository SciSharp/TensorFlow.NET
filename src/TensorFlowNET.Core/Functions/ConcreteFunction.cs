using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Graphs;
using static Tensorflow.Binding;

namespace Tensorflow.Functions
{
    /// <summary>
    /// 
    /// </summary>
    public class ConcreteFunction : IDisposable
    {
        public string Name => _handle == IntPtr.Zero ? string.Empty : c_api.StringPiece(c_api.TF_FunctionName(_handle));
        IntPtr _handle;

        public ConcreteFunction(Func<Tensor, Tensor> func, TF_DataType dtype)
        {
            string func_name = $"autograph_{Guid.NewGuid()}_{func.Method.Name}";

            tf.compat.v1.disable_eager_execution();

            // IntPtr func_handle;
            using (var graph = new FuncGraph(func_name))
            {
                graph.as_default();
                var input = tf.placeholder(dtype);
                var output = func(input);

                var opers = graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
                _handle = graph.ToGraph(opers,
                    new Operation[] { input },
                    new Operation[] { output },
                    null);
            }

            tf.enable_eager_execution();
        }

        public Tensor Execute(Tensor arg)
        {
            var result = tf.Runner.TFE_Execute(tf.Context,
                tf.Context.DeviceName,
                Name,
                new[] { arg },
                null,
                1);
            return result[0];
        }

        public void Dispose()
        {
            c_api.TFE_ContextRemoveFunction(tf.Context.Handle, Name, tf.Status.Handle);
            c_api.TF_DeleteFunction(_handle);
        }
    }
}
