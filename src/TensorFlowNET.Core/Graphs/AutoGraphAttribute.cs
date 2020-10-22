using MethodBoundaryAspect.Fody.Attributes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Eager;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow.Graphs
{
    [AllowChangingInputArguments]
    public sealed class AutoGraphAttribute : OnMethodBoundaryAspect
    {
        FuncGraph graph;
        Tensor[] originalInputs;
        string func_name;
        static Dictionary<string, Func<Tensor[], Tensor>> functions = new Dictionary<string, Func<Tensor[], Tensor>>();

        public override void OnEntry(MethodExecutionArgs args)
        {
            if (args.Instance is TensorFlowOpLayer op)
                func_name = $"autograph_{op.OpType}.{args.Method.Name}";
            else
                func_name = $"autograph_{args.Instance}.{args.Method.Name}";

            if (functions.ContainsKey(func_name))
            {
                args.ReturnValue = functions[func_name](args.Arguments.Select(x => x as Tensor).ToArray());
                args.FlowBehavior = FlowBehavior.Return;
                return;
            }   
            
            // make function as an Operation by autograph
            graph = new FuncGraph(func_name);

            originalInputs = new Tensor[args.Arguments.Length];
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

        public override void OnExit(MethodExecutionArgs args)
        {
            var output = (Tensor)args.ReturnValue;
            var inputs = args.Arguments.Select(x => x as Tensor).ToArray();
            var opers = graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();

            graph.ToGraph(opers,
                inputs.Select(x => x.op).ToArray(),
                new Operation[] { output.op },
                null);

            graph.Dispose();

            Func<Tensor[], Tensor> function = (x) =>
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
