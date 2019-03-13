using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Operations
{
    public abstract class ControlFlowContext : IPython, IControlFlowContext
    {
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
