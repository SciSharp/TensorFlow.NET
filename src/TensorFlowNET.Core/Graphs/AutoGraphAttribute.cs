using MethodBoundaryAspect.Fody.Attributes;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Eager;
using static Tensorflow.Binding;

namespace Tensorflow.Graphs
{
    [AllowChangingInputArguments]
    public sealed class AutoGraphAttribute : OnMethodBoundaryAspect
    {
        FuncGraph graph;
        Tensors originalInputs;
        string func_name;
        static Dictionary<string, Func<Tensors, Tensors>> functions = new Dictionary<string, Func<Tensors, Tensors>>();

        public override void OnEntry(MethodExecutionArgs args)
        {
            func_name = $"autograph_{args.Instance.GetHashCode()}.{args.Method.Name}";

            if (functions.ContainsKey(func_name))
            {
                if (args.Arguments[0] is Tensors tensor_inputs)
                    args.ReturnValue = functions[func_name](tensor_inputs.ToArray());
                else
                    args.ReturnValue = functions[func_name](args.Arguments.Select(x => x as Tensor).ToArray());
                args.FlowBehavior = FlowBehavior.Return;
                return;
            }

            // make function as an Operation by autograph
            graph = new FuncGraph(func_name);

            // convert to Tensors
            if (args.Arguments[0] is Tensors inputs)
            {
                originalInputs = inputs;
                var new_inputs = inputs.Select(x => tf.placeholder(x.dtype, shape: x.TensorShape)).ToArray();
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
                        args.Arguments[i] = tf.placeholder(tensor.dtype, shape: tensor.TensorShape);
                    }
                }
            }
        }

        public override void OnExit(MethodExecutionArgs args)
        {
            var opers = graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();

            if (args.ReturnValue is Tensors outputs)
            {
                if (args.Arguments[0] is Tensors inputs)
                {
                    graph.ToGraph(opers,
                        inputs.Select(x => x.op).ToArray(),
                        outputs.Select(x => x.op).ToArray(),
                        null);
                }
                else
                {
                    graph.ToGraph(opers,
                        args.Arguments.Select(x => (x as Tensor).op).ToArray(),
                        outputs.Select(x => x.op).ToArray(),
                        null);
                }
            }
            else
            {
                graph.ToGraph(opers,
                    args.Arguments.Select(x => (x as Tensor).op).ToArray(),
                    new Operation[] { (args.ReturnValue as Tensor).op },
                    null);
            }

            graph.Dispose();

            Func<Tensors, Tensors> function = (x) =>
            {
                var result = tf.Runner.TFE_Execute(tf.Context,
                    tf.Context.DeviceName,
                    func_name,
                    x,
                    null,
                    1);

                return result[0];
            };
            // cache function.
            functions[func_name] = function;

            // run function
            args.ReturnValue = function(originalInputs);
        }
    }
}
