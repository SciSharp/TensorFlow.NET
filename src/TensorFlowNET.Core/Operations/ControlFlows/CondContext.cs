using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Operations
{
    /// <summary>
    /// The context for the conditional construct.
    /// </summary>
    public class CondContext : ControlFlowContext
    {
        private string _name;

        /// <summary>
        /// The boolean tensor for the cond predicate
        /// </summary>
        private Tensor _pred;
        public Tensor pred => _pred;

        /// <summary>
        /// The predicate tensor in this branch
        /// </summary>
        private Tensor _pivot;

        /// <summary>
        /// 0 or 1 representing this branch
        /// </summary>
        private int _branch;

        /// <summary>
        /// 
        /// </summary>
        private List<string> _values = new List<string>();

        private Dictionary<string, Tensor> _external_values = new Dictionary<string, Tensor>();

        /// <summary>
        /// 
        /// </summary>
        /// <param name="pred">The `boolean` tensor for the conditional predicate.</param>
        /// <param name="pivot">The predicate tensor in this branch.</param>
        /// <param name="branch">0 or 1 representing this branch.</param>
        /// <param name="name">Name of the `CondContext` python object.</param>
        /// <param name="context_def"></param>
        /// <param name="import_scope"></param>
        public CondContext(Tensor pred,
            Tensor pivot,
            int branch,
            string name = "cond_text",
            object context_def = null,
            string import_scope = null)
        {
            _name = ops.get_default_graph().unique_name(name);
            if (context_def != null)
                throw new NotImplementedException("CondContext context_def is not null");
            else
            {
                // Initializes the default fields.
                base.__init__();
                _pred = pred;
                _pivot = pivot;

                // Values considered to have been already seen in this context. pred is not
                // included in this context.
                _values.Add(pred.name);
                _external_values[pred.name] = pred;
                _values.Add(pivot.name);
                pivot.op._set_control_flow_context(this);
            }
        }

        public (T[], Tensor[]) BuildCondBranch<T>(Func<T[]> fn)
        {
            // Add the subgraph defined by fn() to the graph.
            var pre_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION);
            var original_result = fn();
            var post_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION);

            switch (original_result)
            {
                case Tensor[] results:
                    return (original_result, results);
                case float[] fv:
                    var result = ops.convert_to_tensor(fv[0]);
                    return (original_result, new Tensor[] { result });
                default:
                    return (original_result, new Tensor[0]);
            }
        }
    }
}
