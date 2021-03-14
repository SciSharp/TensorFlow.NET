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
    public class ConcreteFunction
    {
        FuncGraph func_graph;
        public Tensor[] Inputs => func_graph.Inputs;
        public Tensor[] CapturedInputs => func_graph.external_captures;

        public string Name => func_graph?.FuncName;

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
            string func_name = $"{func.Method.Name}_{ops.uid_function()}";

            func_graph = new FuncGraph(func_name);
            func_graph.as_default();
            var input = tf.placeholder(dtype);
            var output = func(input);

            var opers = func_graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
            func_graph.ToGraph(opers,
                new[] { input },
                new[] { output },
                null);
            func_graph.Exit();
        }

        public ConcreteFunction(Func<Tensor, IDatasetV2> func, TF_DataType dtype)
        {
            string func_name = $"{func.Method.Name}_{ops.uid_function()}";

            func_graph = new FuncGraph(func_name);
            func_graph.as_default();

            var input = tf.placeholder(dtype);
            var output = func(input);

            OutputStructure = output.structure;

            var opers = func_graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
            func_graph.ToGraph(opers,
                new[] { input },
                new[] { output.variant_tensor },
                null);
            func_graph.Exit();
        }

        public ConcreteFunction(Func<Tensors, Tensors> func,
            TF_DataType[] dtypes, TensorShape[] shapes)
        {
            string func_name = $"{func.Method.Name}_{ops.uid_function()}";

            // IntPtr func_handle;
            func_graph = new FuncGraph(func_name);
            func_graph.as_default();

            var inputs = new Tensors();
            foreach(var (i, dtype) in enumerate(dtypes))
                inputs.Add(tf.placeholder(dtypes[i], shape: shapes[i], name: "args"));
            Outputs = func(inputs);
            OutputStructure = Outputs.Select(x => x.ToTensorSpec()).ToArray();

            var opers = func_graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
            func_graph.ToGraph(opers, inputs, Outputs, null);
            func_graph.Exit();
        }

        public void ToGraph(Tensors inputs, Tensors outputs)
        {
            var opers = func_graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
            func_graph.ToGraph(opers,
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
        public Tensors CallFlat(Tensor[] args, Tensor[] captured_inputs)
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
    }
}
