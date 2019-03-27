using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Operations
{
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

        public void __init__()
        {

        }

        public void __enter__()
        {
        }

        public virtual void Enter()
        {
            var graph = ops.get_default_graph();
            _context_stack.Push(graph._get_control_flow_context());
            graph._set_control_flow_context(this);
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

        public void Exit()
        {
            var graph = ops.get_default_graph();
            var last_context = _context_stack.Pop();
            graph._set_control_flow_context(last_context);
        }

        public void __exit__()
        {
        }

        public void Dispose()
        {
        }
    }
}
