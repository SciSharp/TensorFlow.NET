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

            // If src_graph is a _FuncGraph (i.e. a function body), gather it and all
            // ancestor graphs. This is necessary for correctly handling captured values.
            var curr_graph = src_graph;

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
            {
                grad_scope = namescope;
                // Get a uid for this call to gradients that can be used to help
                // cluster ops for compilation.
                var gradient_uid = ops.get_default_graph().unique_name("uid");

                var to_ops = ys1.Select(x => x.op).ToList();
                var from_ops = xs1.Select(x => x.op).ToList();
                var stop_gradient_ops = stop_gradients1.Select(x => x.op).ToList();
                _PendingCount(to_ops, from_ops, colocate_gradients_with_ops, new List<object>(), xs1);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="grad_ys"></param>
        /// <param name="ys"></param>
        /// <param name="colocate_gradients_with_ops"></param>
        /// <param name="gradient_uid"></param>
        private void _DefaultGradYs(List<Tensor> grad_ys, List<Tensor> ys, bool colocate_gradients_with_ops, string gradient_uid = "__unsupported__")
        {

        }

        /// <summary>
        /// Initialize the pending count for ops between two lists of Operations.
        /// 'pending_count[op]' indicates the number of backprop inputs
        /// to this operation.
        /// </summary>
        /// <param name="to_ops"></param>
        /// <param name="from_ops"></param>
        /// <param name="colocate_gradients_with_ops"></param>
        /// <param name="func_graphs"></param>
        /// <param name="xs"></param>
        private static void _PendingCount(List<Operation> to_ops, List<Operation> from_ops, bool colocate_gradients_with_ops, List<object> func_graphs, List<Tensor> xs)
        {
            List<Operation> reached_ops = new List<Operation>();
            _MarkReachedOps(from_ops, reached_ops, func_graphs);
        }

        /// <summary>
        /// Mark all ops reached from "from_ops"
        /// </summary>
        /// <param name="from_ops"></param>
        /// <param name="reached_ops"></param>
        /// <param name="func_graphs"></param>
        private static void _MarkReachedOps(List<Operation> from_ops, List<Operation> reached_ops, List<object> func_graphs)
        {
            foreach(var op in from_ops)
            {
                reached_ops.Add(op);
                foreach(var output in op.outputs)
                {
                    reached_ops.AddRange(_Consumers(output, func_graphs));
                }
            }

            reached_ops.Reverse();
        }

        /// <summary>
        /// Returns the consumers of t, crossing closure boundaries where necessary.
        /// </summary>
        /// <param name="t"></param>
        /// <param name="func_graphs"></param>
        private static List<Operation> _Consumers(Tensor t, List<object> func_graphs)
        {
            var consumers = t.consumers();
            return consumers;
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
