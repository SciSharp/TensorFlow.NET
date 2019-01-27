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
    }
}
