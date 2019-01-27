using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;

namespace Tensorflow
{
    /// <summary>
    /// Context manager for `control_dependencies()`
    /// </summary>
    public class _ControlDependenciesController : IPython
    {
        private Graph _graph;
        private List<Operation> _control_inputs_val;
        private List<Operation> _seen_nodes;
        private Queue<_ControlDependenciesController> _old_stack;
        private bool _new_stack;
        private Context _old_control_flow_context;

        public Operation[] control_inputs => _control_inputs_val.ToArray();

        public _ControlDependenciesController(Graph graph, List<Operation> control_inputs)
        {
            _graph = graph;
            if (control_inputs == null)
            {
                _control_inputs_val = new List<Operation>();
                _new_stack = true;
            }
            else
            {
                _control_inputs_val = control_inputs;
                _new_stack = false;
            }

            _seen_nodes = new List<Operation>();
        }

        public void add_op(Operation op)
        {
            _seen_nodes.Add(op);
        }

        public bool op_in_group(Operation op)
        {
            return _seen_nodes.Contains(op);
        }

        public void __enter__()
        {
            if (_new_stack)
            {
                // Clear the control_dependencies graph.
                _old_stack = _graph._control_dependencies_stack;
                _graph._control_dependencies_stack = new Queue<_ControlDependenciesController>();

                // Clear the control_flow_context too.
                _old_control_flow_context = _graph._get_control_flow_context();
                _graph._set_control_flow_context(null);
            }

            _graph._push_control_dependencies_controller(this);
        }

        public void __exit__()
        {
            _graph._pop_control_dependencies_controller(this);
            if (_new_stack)
            {
                _graph._control_dependencies_stack = _old_stack;
                _graph._set_control_flow_context(_old_control_flow_context);
            }
        }

        public void Dispose()
        {
            
        }
    }
}
