using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public interface IControlFlowContext
    {
        void AddOp(Operation op);
    }
}
