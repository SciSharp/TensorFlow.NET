/*using MethodBoundaryAspect.Fody.Attributes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Eager;
using static Tensorflow.Binding;

namespace Tensorflow.Graphs
{
    public sealed class AutoGraphAspect : OnMethodBoundaryAspect
    {
        FuncGraph graph;
        IntPtr func_handle;

        public override void OnEntry(MethodExecutionArgs args)
        {
            tf.compat.v1.disable_eager_execution();
            // convert args to placeholder
            
            for (var i = 0; i < args.Arguments.Length; i++)
            {
                if (args.Arguments[i] is EagerTensor tensor)
                    args.Arguments[i] = tf.placeholder(tensor.dtype, shape: tensor.TensorShape);
            }

            // make function as an Operation by autograph
            graph = new FuncGraph("autograph_add");
            graph.as_default();
        }

        public override void OnExit(MethodExecutionArgs args)
        {
            var output = (Tensor)args.Method.Invoke(args.Instance, args.Arguments);
            var opers = graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
            func_handle = graph.ToGraph(opers,
                new Operation[] {  },
                new Operation[] {  },
                null);


            c_api.TFE_ContextAddFunction(tf.Context.Handle, func_handle, tf.Status.Handle);

            var a1 = tf.constant(1);
            var b1 = tf.constant(2);

            var result = tf.Runner.TFE_Execute(tf.Context,
                tf.Context.DeviceName,
                "autograph_add",
                new[] { a1, b1 },
                null,
                1);
            graph.Dispose();
        }
    }
}*/
