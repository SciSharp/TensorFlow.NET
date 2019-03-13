using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations;

namespace Tensorflow
{
    public partial class Operation
    {
        private CondContext _control_flow_context;

        /// <summary>
        /// Add this op to its control flow context.
        /// </summary>
        public void _control_flow_post_processing()
        {
            foreach(var input_tensor in inputs)
            {

            }
        }

        public void _add_control_inputs(Operation[] ops)
        {
            foreach(var op in ops)
            {
                c_api.TF_AddControlInput(graph, op);
            }
        }

        public void _set_control_flow_context(CondContext ctx)
        {
            _control_flow_context = ctx;
        }
    }
}
