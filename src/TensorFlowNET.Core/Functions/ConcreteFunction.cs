using System;
using System.Collections.Generic;
using System.Linq;
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
        IntPtr _handle;
        FuncGraph func_graph;

        public string Name
        {
            get
            {
                if (func_graph != null)
                    return func_graph.FuncName;

                return _handle == IntPtr.Zero ? string.Empty : c_api.StringPiece(c_api.TF_FunctionName(_handle));
            }
        }
        
        public Tensor[] Outputs;
        public Type ReturnType;
        public TensorSpec[] OutputStructure;

        public ConcreteFunction(string name)
        {
            func_graph = new FuncGraph(name);
        }

        public ConcreteFunction(FuncGraph graph, Dictionary<string, string> attrs)
        {
            func_graph = graph;
        }

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
                    new[] { input },
                    new[] { output },
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
                    new[] { input },
                    new[] { output.variant_tensor },
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
                    new[] { input1, input2, input3 },
                    new[] { outputs.Item1, outputs.Item2 },
                    null);
            }
        }

        public void ToGraph(Tensors inputs, Tensors outputs)
        {
            var opers = func_graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
            _handle = func_graph.ToGraph(opers,
                inputs,
                outputs,
                null);

            OutputStructure = outputs.Select(x => x.ToTensorSpec()).ToArray();
        }

        public Tensors Invoke(Tensors inputs)
        {
            var forward_backward = SelectForwardAndBackwardFunctions(inputs, 1, tf.Context.executing_eagerly());
            var (forward_function, args_with_tangents) = forward_backward.Forward();
            Tensors flat_outputs = null;
            if (tf.Context.executing_eagerly())
                flat_outputs = forward_function.Call(args_with_tangents);
            forward_backward.Record(flat_outputs);
            return flat_outputs;
        }

        ForwardBackwardCall SelectForwardAndBackwardFunctions(Tensors args, int possible_gradient_type, bool executing_eagerly)
        {
            var functions = new FirstOrderTapeGradientFunctions(func_graph, false);
            return new ForwardBackwardCall(functions, args, tape_watching: true);
        }

        public override string ToString()
            => Name;

        public void Dispose()
        {
            c_api.TFE_ContextRemoveFunction(tf.Context.Handle, Name, tf.Status.Handle);
            c_api.TF_DeleteFunction(_handle);
        }
    }
}
