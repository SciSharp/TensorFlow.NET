using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations.ControlFlows;

namespace Tensorflow.Operations
{
    public class WhileContext : ControlFlowContext
    {
        private bool _back_prop=true;

        private GradLoopState _grad_state =null;

        public override WhileContext GetWhileContext()
        {
            return this;
        }


        public override GradLoopState grad_state => _grad_state;

        public override bool back_prop => _back_prop;

        public static WhileContext from_proto(object proto)
        {
            throw new NotImplementedException();
        }

        public object to_proto()
        {
            throw new NotImplementedException();
        }
    }
}
