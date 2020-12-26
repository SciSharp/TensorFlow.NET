using MethodBoundaryAspect.Fody.Attributes;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Eager;
using Tensorflow.Functions;
using static Tensorflow.Binding;

namespace Tensorflow.Graphs
{
    [AllowChangingInputArguments]
    public sealed class AutoGraphAttribute : OnMethodBoundaryAspect
    {
        ConcreteFunction function;
        Tensors originalInputs;
        string func_name;
        static Dictionary<string, ConcreteFunction> functions = new Dictionary<string, ConcreteFunction>();

        public override void OnEntry(MethodExecutionArgs args)
        {
            func_name = $"autograph_{args.Instance.GetType().FullName}.{args.Method.Name}";

            if (functions.ContainsKey(func_name))
            {
                function = functions[func_name];
                if (args.Arguments[0] is Tensors tensor_inputs)
                    args.ReturnValue = ConvertReturnValue(function.Invoke(tensor_inputs));
                else
                    args.ReturnValue = ConvertReturnValue(function.Invoke(args.Arguments.Select(x => x as Tensor).ToArray()));
                args.FlowBehavior = FlowBehavior.Return;
                return;
            }

            // make function as an Operation by autograph
            // need to restore mode when exits
            function = new ConcreteFunction(func_name);

            // convert to Tensors
            if (args.Arguments[0] is Tensors inputs)
            {
                originalInputs = inputs;
                var new_inputs = inputs.Select(x => tf.placeholder(x.dtype, shape: x.TensorShape, name: "inputs")).ToArray();
                args.Arguments[0] = new Tensors(new_inputs);
            }
            else
            {
                originalInputs = new Tensors(args.Arguments.Length);
                // convert args to placeholder
                for (var i = 0; i < args.Arguments.Length; i++)
                {
                    if (args.Arguments[i] is EagerTensor tensor)
                    {
                        originalInputs[i] = tensor;
                        args.Arguments[i] = tf.placeholder(tensor.dtype, shape: tensor.TensorShape, name: "inputs");
                    }
                }
            }
        }

        public override void OnExit(MethodExecutionArgs args)
        {
            if (args.ReturnValue is Tensors outputs)
            {
                if (args.Arguments[0] is Tensors inputs)
                    function.ToGraph(inputs, outputs);
                else
                    function.ToGraph(args.Arguments.Select(x => x as Tensor).ToArray(), outputs);
            }
            else
                function.ToGraph(args.Arguments.Select(x => x as Tensor).ToArray(), args.ReturnValue as Tensor);

            // cache function.
            function.ReturnType = args.ReturnValue.GetType();
            functions[func_name] = function;

            // run function
            args.ReturnValue = ConvertReturnValue(function.Invoke(originalInputs));
        }

        object ConvertReturnValue(Tensors tensors)
        {
            if (function.ReturnType == typeof(Tensor))
                return (Tensor)tensors;
            else
                return tensors;
        }
    }
}
