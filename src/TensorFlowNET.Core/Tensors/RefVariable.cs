using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class RefVariable : Variable
    {
        public bool _in_graph_mode = true;

        public RefVariable(object initial_value, 
            TF_DataType trainable, 
            bool validate_shape = true) : 
            base(initial_value, trainable, validate_shape)
        {

        }

        private void _init_from_args()
        {

        }
    }
}
