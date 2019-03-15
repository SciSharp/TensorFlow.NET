using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Gradients
{
    public class control_flow_grad
    {
        public static Tensor[] _MergeGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var _ = grads[1];
            var input_op = op.inputs[0].op;
            var graph = ops.get_default_graph();
            var op_ctxt = control_flow_util.GetOutputContext(input_op);
            var pred = op_ctxt.pred;

            var results = control_flow_ops._SwitchRefOrTensor(grad, pred, name: "cond_grad");
            return new Tensor[] { results.Item1, results.Item2 };
        }
    }
}
