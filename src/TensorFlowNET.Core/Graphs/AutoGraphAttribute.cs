using MethodBoundaryAspect.Fody.Attributes;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Eager;
using Tensorflow.Functions;
using static Tensorflow.Binding;

namespace Tensorflow.Graphs
{
    /// <summary>
    /// func_graph.py func_graph_from_py_func
    /// </summary>
    [AllowChangingInputArguments]
    public sealed class AutoGraphAttribute : OnMethodBoundaryAspect
    {
        ConcreteFunction function;
        Tensors originalInputs;
        string func_name;
        static Dictionary<string, ConcreteFunction> functions = new Dictionary<string, ConcreteFunction>();

        public override void OnEntry(MethodExecutionArgs args)
        {
            // TODO: func_name can be cache in FullName + Args
            func_name = $"{args.Method.DeclaringType.FullName}.{args.Method.Name}_{Guid.NewGuid()}";

            if (functions.ContainsKey(func_name))
            {
                function = functions[func_name];
                if (args.Arguments[0] is Tensors tensor_inputs)
                    args.ReturnValue = ConvertReturnValue(function.FilteredCall(tensor_inputs));
                else
                    args.ReturnValue = ConvertReturnValue(function.FilteredCall(args.Arguments.Select(x => x as Tensor).ToArray()));
                args.FlowBehavior = FlowBehavior.Return;
                return;
            }

            // make function as an Operation by autograph
            // need to restore mode when exits
            function = new ConcreteFunction(func_name);
            function.Enter();

            // convert to Tensors
            if (args.Arguments[0] is Tensors inputs)
            {
                originalInputs = inputs;
                var new_inputs = inputs.Select(x => tf.placeholder(x.dtype, shape: x.TensorShape, name: "inputs")).ToArray();
                args.Arguments[0] = new Tensors(new_inputs);
            }
            else
            {
                originalInputs = new Tensors();
                // convert args to placeholder
                for (var i = 0; i < args.Arguments.Length; i++)
                {
                    if (args.Arguments[i] is EagerTensor tensor)
                    {
                        originalInputs.Add(tensor);
                        args.Arguments[i] = tf.placeholder(tensor.dtype, shape: tensor.TensorShape, name: "inputs");
                    }
                }
            }
        }

        public override void OnExit(MethodExecutionArgs args)
        {
            if (args.ReturnValue is Tensors outputs)
            {
                Tensors inputs = null;
                outputs = mark_as_return(outputs);
                if (args.Arguments[0] is Tensors inputs1)
                    inputs = inputs1;
                else
                    inputs = args.Arguments.Select(x => x as Tensor).ToArray();

                inputs = inputs.Where(x => x.op.OpType == "Placeholder" 
                    && x.op.name.StartsWith("inputs")).ToArray();

                function.ToGraph(inputs, outputs);
            }
            else if (args.ReturnValue is Tensor output)
            {
                var inputs = args.Arguments.Select(x => x as Tensor)
                    .Where(x => x.op.type == "Placeholder" && x.op.name.StartsWith("inputs"))
                    .ToArray();
                var outputs2 = array_ops.identity(output);
                function.ToGraph(inputs, outputs2);
            }

            function.Exit();

            // cache function.
            function.ReturnType = args.ReturnValue.GetType();
            functions[func_name] = function;

            // run function
            args.ReturnValue = ConvertReturnValue(function.FilteredCall(originalInputs));
        }

        object ConvertReturnValue(Tensors tensors)
        {
            if (function.ReturnType == typeof(Tensor))
                return (Tensor)tensors;
            else
                return tensors;
        }

        /// <summary>
        /// Acts like identity but marks the `Tensor` as a return value.
        /// </summary>
        /// <param name="tensors"></param>
        /// <returns></returns>
        public Tensors mark_as_return(Tensors tensors)
        {
            if (tensors == null)
                return null;
            var result = new Tensors();
            foreach (var tensor in tensors)
                result.Add(array_ops.identity(tensor));
            return result;
        }
    }
}
