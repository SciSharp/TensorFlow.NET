using System;
using System.Collections.Generic;
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
    public abstract class ControlFlowContext : IPython, IControlFlowContext
    {
        /// <summary>
        /// The predicate tensor in this branch
        /// </summary>
        protected Tensor _pivot;

        protected Stack<IControlFlowContext> _context_stack;
        public ControlFlowContext()
        {
            _context_stack = new Stack<IControlFlowContext>();
        }

        public string name { get; set; }

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

        public void AddOp(Operation op)
        {
            _AddOpInternal(op);
        }

        protected virtual void _AddOpInternal(Operation op)
        {
            if(op.inputs.Length == 0)
            {
                _RemoveExternalControlEdges(op);
                op._add_control_input(_pivot.op);
            }
            else
            {

            }
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
