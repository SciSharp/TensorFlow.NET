using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Framework.Models;
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
        public Tensor[] Outputs;
        public TensorSpec[] OutputStructure;

        public ConcreteFunction(Func<Tensor, Tensor> func, TF_DataType dtype)
        {
            string func_name = $"autograph_{Guid.NewGuid()}_{func.Method.Name}";

            // IntPtr func_handle;
            using (var graph = new FuncGraph(func_name))
            {
                var input = tf.placeholder(dtype);
                var output = func(input);

                var opers = graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
                _handle = graph.ToGraph(opers,
                    new Operation[] { input },
                    new Operation[] { output },
                    null);
            }
        }

        public ConcreteFunction(Func<Tensor, IDatasetV2> func, TF_DataType dtype)
        {
            string func_name = $"autograph_{Guid.NewGuid()}_{func.Method.Name}";

            // IntPtr func_handle;
            using (var graph = new FuncGraph(func_name))
            {
                var input = tf.placeholder(dtype);
                var output = func(input);

                OutputStructure = output.structure;

                var opers = graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
                _handle = graph.ToGraph(opers,
                    new Operation[] { input },
                    new Operation[] { output.variant_tensor.op },
                    null);
            }
        }

        public ConcreteFunction(Func<Tensor, (Tensor, Tensor), (Tensor, Tensor)> func, 
            TF_DataType[] dtypes, TensorShape[] shapes)
        {
            string func_name = $"autograph_{Guid.NewGuid()}_{func.Method.Name}";

            // IntPtr func_handle;
            using (var graph = new FuncGraph(func_name))
            {
                var input1 = tf.placeholder(dtypes[0], shape: shapes[0], name: "args");
                var input2 = tf.placeholder(dtypes[1], shape: shapes[1], name: "args");
                var input3 = tf.placeholder(dtypes[2], shape: shapes[2], name: "args");
                var outputs = func(input1, (input2, input3));

                Outputs = new[] { outputs.Item1, outputs.Item2 };
                OutputStructure = new[] { outputs.Item1.ToTensorSpec(), outputs.Item2.ToTensorSpec() };

                var opers = graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
                _handle = graph.ToGraph(opers,
                    new Operation[] { input1, input2, input3 },
                    new Operation[] { outputs.Item1.op, outputs.Item2.op },
                    null);
            }
        }

        public void Dispose()
        {
            c_api.TFE_ContextRemoveFunction(tf.Context.Handle, Name, tf.Status.Handle);
            c_api.TF_DeleteFunction(_handle);
        }
    }
}
