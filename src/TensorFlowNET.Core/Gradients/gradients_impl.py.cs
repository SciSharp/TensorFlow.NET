using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class gradients_impl
    {
        public static void gradients(object ys,
            object xs, 
            List<Tensor> grad_ys = null,
            string name = "gradients",
            bool colocate_gradients_with_ops = false,
            bool gate_gradients = false,
            int? aggregation_method = null)
        {
            _GradientsHelper(ys, xs, grad_ys, name, colocate_gradients_with_ops, gate_gradients);
        }

        public static void _GradientsHelper(object ys,
            object xs,
            List<Tensor> grad_ys = null,
            string name = "gradients",
            bool colocate_gradients_with_ops = false,
            bool gate_gradients = false,
            Graph src_graph = null)
        {
            if (src_graph == null)
                src_graph = ops.get_default_graph();
        }
    }
}
