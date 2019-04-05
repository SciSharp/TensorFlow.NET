using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations;

namespace Tensorflow
{
    public partial class Operation
    {
        private IControlFlowContext _control_flow_context;

        /// <summary>
        /// Add this op to its control flow context.
        /// </summary>
        public void _control_flow_post_processing()
        {
            foreach(var input_tensor in inputs)
            {

            }

            if (_control_flow_context != null)
                _control_flow_context.AddOp(this);
        }

        public void _add_control_input(Operation op)
        {
            c_api.TF_AddControlInput(_operDesc, op);
        }

        public void _add_control_inputs(Operation[] ops)
        {
            foreach (var op in ops)
                _add_control_input(op);
        }

        public void _set_control_flow_context(IControlFlowContext ctx)
        {
            _control_flow_context = ctx;
        }

        public IControlFlowContext _get_control_flow_context()
        {
            return _control_flow_context;
        }
    }
}
