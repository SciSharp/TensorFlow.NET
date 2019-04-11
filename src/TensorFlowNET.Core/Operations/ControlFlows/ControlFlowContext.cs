using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow.Operations
{
    /// <summary>
    /// The base class for control flow context.
    /// 
    /// The usage pattern is a sequence of(Enter, Exit) followed by a final
    /// ExitResult.
    /// 
    /// We maintain the following state for control flow contexts during graph
    /// construction:
    /// 1. graph has _control_flow_context: the current context used to
    /// construct new nodes.Changed by ctxt.Enter() and ctxt.Exit()
    /// 2. op has _control_flow_context: the context to which the op belongs.
    /// Set at the time the op is created.Immutable.
    /// 3. A ControlFlowContext has _outer_context: the context in which this
    /// context is created.Set at the time a context is created.Immutable.
    /// 4. A ControlFlowContext has _context_stack.
    /// Pushed and popped by ctxt.Enter() and ctxt.Exit()
    /// </summary>
    public abstract class ControlFlowContext : Python, IPython, IControlFlowContext
    {
        /// <summary>
        /// The predicate tensor in this branch
        /// </summary>
        protected Tensor _pivot;

        protected Stack<IControlFlowContext> _context_stack;
        protected IControlFlowContext _outer_context;

        public ControlFlowContext()
        {
            _context_stack = new Stack<IControlFlowContext>();
        }

        public string name { get => _name; }
        protected string _name;

        public void __init__()
        {

        }

        public void __enter__()
        {
        }

        public void __exit__()
        {
        }

        /// <summary>
        /// Enter this control flow context.
        /// </summary>
        public virtual void Enter()
        {
            var graph = ops.get_default_graph();
            _context_stack.Push(graph._get_control_flow_context());
            graph._set_control_flow_context(this);
        }

        /// <summary>
        /// Exit this control flow context.
        /// </summary>
        public virtual void Exit()
        {
            var graph = ops.get_default_graph();
            var last_context = _context_stack.Pop();
            graph._set_control_flow_context(last_context);
        }

        /// <summary>
        /// Add `op` to the current context.
        /// </summary>
        public void AddOp(Operation op)
        {
            _AddOpInternal(op);
        }

        public IControlFlowContext outer_context { get { return _outer_context; } }
        public HashSet<string> values => _values;
        public virtual Tensor AddValue(Tensor val)
        {
            // to be overridden
            return null;
        }

        public virtual void AddInnerOp(Operation resultOp)
        {
            // to be overridden
        }

        protected HashSet<string> _values = new HashSet<string>();

        /// <summary>
        /// Add `op` to the current context.
        /// </summary>
        protected virtual void _AddOpInternal(Operation op)
        {
            if (op.inputs.Length == 0)
            {
                //If we're in a while loop, remove any control inputs from outside the
                // loop.
                _RemoveExternalControlEdges(op);
                if (!op.control_inputs.Any(input_op => OpInContext(input_op)))
                    op._add_control_input(_pivot.op);
            }
            else
            {
                // Make each input to 'op' available in this CondContext. If an input is
                // already part of this context there's nothing to do, but if it's
                // external, AddValue() will handle adding the appropriate Switch node and
                // other bookkeeping.
                for (int index = 0; index < op.inputs.Length; index++)
                {
                    var x = op.inputs[index];
                    Tensor real_x = null;
                    if (op.type == "Merge" && x.op.type == "NextIteration")
                    {
                        //# Edge case: if we're importing a while loop inside this CondContext,
                        //# AddValue() will not correctly handle the NextIteration inputs to
                        //# Merge node. The problem is that the NextIteration should also be
                        //# part of this context, but if we're importing it won't have been
                        //# processed and added to the context yet, so AddValue() will try to
                        //# add a Switch which results in an invalid graph. Instead, we use the
                        //# NextIteration input as-is here, and it will eventually be added to
                        //# the context via AddOp().
                        real_x = x;
                    }
                    else
                    {
                        real_x = AddValue(x);
                    }
                    if (real_x != x)
                        op._update_input(index, real_x);
                }
                // Remove any external control dependency on this op.
                _RemoveExternalControlEdges(op);
                // TODO: implement below code dependencies
                //if (op.graph._is_function(op.type) || op.type == "SymbolicGradient")
                //    op._add_control_input(_pivot.op);
            }
            
            // Mark op's outputs as seen by this context and any outer contexts.
            var output_names = op.outputs.Select(x => x.name).ToArray();
            IControlFlowContext ctxt = this;
            while (ctxt != null)
            {
                foreach(var name in output_names)
                    ctxt.values.Add(name);
                ctxt = ctxt.outer_context;
            }

            if (_outer_context != null || !control_flow_ops.IsLoopExit(op))
                op.graph.prevent_fetching(op);

            if (_outer_context != null)
                _outer_context.AddInnerOp(op);
        }

        private bool OpInContext(Operation op)
        {
            return IsContainingContext(op._get_control_flow_context(), this);
        }

        /// <summary>
        /// Returns true if `maybe_containing_ctxt` is or contains `ctxt`.
        /// </summary>
        public static bool IsContainingContext(IControlFlowContext ctxt, ControlFlowContext maybe_containing_ctxt)
        {
            while (ctxt != maybe_containing_ctxt)
            {
                if (ctxt == null)
                    return false;
                ctxt = ctxt.outer_context;
            }
            return true;
        }


        protected virtual void _RemoveExternalControlEdges(Operation op)
        {
            var internal_control_inputs = op.control_inputs;
        }

        public void Dispose()
        {
        }
    }
}
