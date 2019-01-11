using System;
using System.Collections.Generic;
using System.Linq;
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
            object grad_ys = null,
            string name = "gradients",
            bool colocate_gradients_with_ops = false,
            bool gate_gradients = false,
            object stop_gradients = null,
            Graph src_graph = null)
        {
            if (src_graph == null)
                src_graph = ops.get_default_graph();

            var ys1 = _AsList(ys);
            var xs1 = _AsList(xs);
            List<Tensor> grad_ys1 = null;
            List<Tensor> stop_gradients1 = stop_gradients == null ? new List<Tensor>() : _AsList(stop_gradients);
            if (grad_ys == null)
                grad_ys1 = ys1.Select(x => new Tensor(IntPtr.Zero)).ToList();
            else
                grad_ys = _AsList(grad_ys);

            var all = new List<Tensor>();
            all.AddRange(ys1);
            all.AddRange(xs1);
            all.AddRange(stop_gradients1);
            all.AddRange(grad_ys1);

            string grad_scope = "";
            using (var namescope = new ops.name_scope<Tensor>(name, "gradients", values: all))
                grad_scope = namescope;
        }

        private static List<Tensor> _AsList(object ys)
        {
            List<Tensor> ret = null;

            switch (ys)
            {
                case Tensor value:
                    ret = new List<Tensor> { value };
                    break;
                case List<Tensor> value:
                    ret = value;
                    break;
            }

            return ret;
        }
    }
}
