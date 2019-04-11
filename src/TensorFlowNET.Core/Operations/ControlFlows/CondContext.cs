using System;
using System.Collections.Generic;
using System.Linq;
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
        /// 0 or 1 representing this branch
        /// </summary>
        private int _branch;

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

        /// <summary>
        /// Add `val` to the current context and its outer context recursively.
        /// </summary>
        /// <param name="val"></param>
        public override Tensor AddValue(Tensor val)
        {
            Tensor result = null;
            if (_values.Contains(val.name))
            {
                // Use the real value if it comes from outer context. This is needed in
                // particular for nested conds.
                if (_external_values.ContainsKey(val.name))
                    result = _external_values[val.name];
                else
                    result = val;
            }
            else
            {
                result = val;
                _values.Add(val.name);
                // TODO: _outer_context                
                if (_outer_context != null)
                {
                    result = _outer_context.AddValue(val);
                    _values.Add(result.name);
                    _external_values[result.name] = result;
                }
                
                with(ops.control_dependencies(null), ctrl =>
                {
                    var (r0, r1) = control_flow_ops._SwitchRefOrTensor(result, _pred);
                    result = new[] { r0, r1 }[_branch];
                    if (_outer_context != null)
                        _outer_context.AddInnerOp(result.op);
                });

                result.op.graph.prevent_fetching(result.op);
                result.op._set_control_flow_context(this);

                // Mark Switch output as seen by this context and any outer contexts,
                // just like what we do for normal op outputs in _AddOpInternal() below.
                IControlFlowContext ctxt = this;
                while (ctxt != null)
                {
                    ctxt.values.Add(result.name);
                    ctxt = ctxt.outer_context;
                }
                _external_values[val.name] = result;
            }
            return result;
        }

        /// <summary>
        /// Add the subgraph defined by fn() to the graph.
        /// </summary>
        public (T, Tensor) BuildCondBranch<T>(Func<T> fn)
        {
            // Add the subgraph defined by fn() to the graph.
            var pre_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION);
            var original_result = fn();
            var post_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION);

            //TODO: port this chunck of missing code:
            /*
            if len(post_summaries) > len(pre_summaries):
                new_summaries = post_summaries[len(pre_summaries):]
                summary_ref = ops.get_collection_ref(ops.GraphKeys._SUMMARY_COLLECTION)  # pylint: disable=protected-access
                summary_ref[:] = pre_summaries
                with ops.control_dependencies(new_summaries):
                if original_result is None:
                    return no_op(), None
                else:
                    original_result = nest.map_structure(array_ops.identity,
                                                        original_result)
            */
            if (original_result == null)
                return (original_result, null);

            switch (original_result)
            {
                case Tensor result:
                    return (original_result, _BuildCondTensor(result));
                case Operation op:
                    return (original_result, _BuildCondTensor(op));
                case float[] fv:
                    {
                        var result = ops.convert_to_tensor(fv[0]);
                        return (original_result, _BuildCondTensor(result));
                    }
                default:
                    return (original_result, null);
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
                    return (original_result, results.Select(_BuildCondTensor).ToArray());
                case Operation[] results:
                    return (original_result, results.Select(_BuildCondTensor).ToArray());
                case float[] fv:
                    var result = ops.convert_to_tensor(fv[0]);
                    return (original_result, new Tensor[] { result });
                default:
                    return (original_result, new Tensor[0]);
            }
        }

        private Tensor _BuildCondTensor(ITensorOrOperation v)
        {
            switch (v)
            {
                case Operation op:
                    // Use pivot as the proxy for this op.
                    return control_flow_ops.with_dependencies(new Operation[] { op }, _pivot);
                case Tensor t:
                    return _ProcessOutputTensor(t);
                default:
                    return _ProcessOutputTensor(ops.convert_to_tensor(v));

            }
        }

        /// <summary>
        /// Process an output tensor of a conditional branch.
        /// </summary>
        private Tensor _ProcessOutputTensor(Tensor val)
        {
            var real_val = val;
            if (!_values.Contains(val.name))
            {
                // Handle the special case of lambda: x
                _values.Add(val.name);
                if (_outer_context != null)
                {
                    real_val = _outer_context.AddValue(val);
                    _values.Add(real_val.name);
                    _external_values[real_val.name] = real_val;
                }
                var (t0, t1) = control_flow_ops._SwitchRefOrTensor(real_val, _pred);
                real_val = new[] {t0, t1}[_branch];
                _external_values[val.name] = real_val;
            }
            else
            {
                Tensor external_val = null;
                if (_external_values.ContainsKey(val.name))
                    external_val = _external_values[val.name];
                if (external_val != null)
                    real_val = external_val;
            }
            return real_val;
        }

        public override void  AddInnerOp(Operation resultOp)
        {
            throw new NotImplementedException();
        }
    }
}