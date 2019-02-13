using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public partial class Operation
    {
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
    }
}
