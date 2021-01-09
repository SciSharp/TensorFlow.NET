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
        public Tensor[] Inputs => func_graph.Inputs;
        public Tensor[] CapturedInputs => func_graph.external_captures;

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

        public ConcreteFunction(FuncGraph graph, Dictionary<string, string> attrs = null)
        {
            func_graph = graph;

            ToGraph(graph.Inputs, graph.Outputs.Where(x => x != null).ToArray());
        }

        public ConcreteFunction(Func<Tensor, Tensor> func, TF_DataType dtype)
        {
            string func_name = $"{func.Method.Name}_{Guid.NewGuid()}";

            // IntPtr func_handle;
            using var graph = new FuncGraph(func_name);
            graph.as_default();
            var input = tf.placeholder(dtype);
            var output = func(input);

            var opers = graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
            _handle = graph.ToGraph(opers,
                new[] { input },
                new[] { output },
                null);
            graph.Exit();
        }

        public ConcreteFunction(Func<Tensor, IDatasetV2> func, TF_DataType dtype)
        {
            string func_name = $"{func.Method.Name}_{Guid.NewGuid()}";

            // IntPtr func_handle;
            using var graph = new FuncGraph(func_name);
            graph.as_default();

            var input = tf.placeholder(dtype);
            var output = func(input);

            OutputStructure = output.structure;

            var opers = graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
            _handle = graph.ToGraph(opers,
                new[] { input },
                new[] { output.variant_tensor },
                null);
            graph.Exit();
        }

        public ConcreteFunction(Func<Tensor, (Tensor, Tensor), (Tensor, Tensor)> func,
            TF_DataType[] dtypes, TensorShape[] shapes)
        {
            string func_name = $"{func.Method.Name}_{Guid.NewGuid()}";

            // IntPtr func_handle;
            using var graph = new FuncGraph(func_name);
            graph.as_default();

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
            graph.Exit();
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

        public void Enter()
        {
            func_graph.as_default();
        }

        public void Exit()
        {
            func_graph.Exit();
        }

        public Tensors FilteredCall(Tensors inputs)
        {
            return CallFlat(inputs, CapturedInputs);
        }

        /// <summary>
        /// Executes the wrapped function.
        /// </summary>
        /// <param name="args"></param>
        /// <param name="captured_inputs"></param>
        /// <returns></returns>
        public Tensor[] CallFlat(Tensor[] args, Tensor[] captured_inputs)
        {
            var executing_eagerly = tf.Context.executing_eagerly();
            var default_graph = ops.get_default_graph();
            var tensor_inputs = new Tensors();
            foreach (var (i, arg) in enumerate(args))
            {
                tensor_inputs.Add(arg);
                // If we're graph building, shape inference is on.
                if (!executing_eagerly)
                {
                }
            }
            tensor_inputs.AddRange(captured_inputs);

            args = tensor_inputs.ToArray();

            var possible_gradient_type = tf.Runner.MustRecordGradient() ? 1 : 0;
            // No tape is watching; skip to running the function.
            if (possible_gradient_type == 0 && executing_eagerly)
            {
                var attrs = new object[]
                {
                    "executor_type", "",
                    "config_proto", tf.Context.FunctionCallOptions.config_proto_serialized()
                };
                return tf.Runner.Execute(tf.Context, func_graph.FuncName, func_graph.Outputs.Length, args, attrs);
            }

            var forward_backward = SelectForwardAndBackwardFunctions(args, possible_gradient_type, executing_eagerly);
            var (forward_function, args_with_tangents) = forward_backward.Forward();
            Tensors flat_outputs = null;
            if (executing_eagerly)
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
