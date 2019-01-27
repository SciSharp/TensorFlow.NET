using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Eager;

namespace Tensorflow
{
    public partial class Graph
    {
        public Context _control_flow_context;

        private Queue<_ControlDependenciesController> _graph_control_dependencies_stack = new Queue<_ControlDependenciesController>();
        public Queue<_ControlDependenciesController> _control_dependencies_stack
        {
            get
            {
                return _graph_control_dependencies_stack;
            }
            set
            {
                _graph_control_dependencies_stack = value;
            }
        }

        /// <summary>
        /// For an op that takes `input_ops` as inputs, compute control inputs.
        /// </summary>
        /// <param name="input_ops">The data input ops for an op to be created.</param>
        /// <returns>A list of control inputs for the op to be created.</returns>
        private Operation[] _control_dependencies_for_inputs(Operation[] input_ops)
        {
            Operation[] ret = new Operation[0];

            foreach(var controller in _control_dependencies_stack)
            {
                bool dominated = false;
                // If any of the input_ops already depends on the inputs from controller,
                // we say that the new op is dominated (by that input), and we therefore
                // do not need to add control dependencies for this controller's inputs.
                foreach(var op in input_ops)
                {
                    if (controller.op_in_group(op))
                    {
                        dominated = true;
                        break;
                    }
                }

                if (!dominated)
                    ret = controller.control_inputs.Where(x => !input_ops.Contains(x)).ToArray();
            }

            return ret;
        }

        public _ControlDependenciesController control_dependencies(Operation[] control_inputs)
        {
            if (control_inputs == null)
                return new _ControlDependenciesController(this, null);

            var control_ops = new List<Operation>();
            foreach (var c in control_inputs)
            {
                control_ops.Add(c);
            }

            return new _ControlDependenciesController(this, control_ops);
        }

        /// <summary>
        /// Returns the current control flow context.
        /// </summary>
        /// <returns>A context object.</returns>
        public Context _get_control_flow_context()
        {
            return _control_flow_context;
        }

        /// <summary>
        /// Sets the current control flow context.
        /// </summary>
        /// <param name="ctx">a context object.</param>
        public void _set_control_flow_context(Context ctx)
        {
            _control_flow_context = ctx;
        }

        public void _push_control_dependencies_controller(_ControlDependenciesController controller)
        {
            _control_dependencies_stack.Enqueue(controller);
        }

        public void _pop_control_dependencies_controller(_ControlDependenciesController controller)
        {
            _control_dependencies_stack.Dequeue();
        }

        public void _record_op_seen_by_control_dependencies(Operation op)
        {
            foreach (var controller in _control_dependencies_stack)
                controller.add_op(op);
        }
    }
}
