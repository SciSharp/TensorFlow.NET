using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public interface IControlFlowContext
    {
        void AddOp(Operation op);
        IControlFlowContext outer_context { get;  }
        HashSet<string> values { get; }
        Tensor AddValue(Tensor val);
        void AddInnerOp(Operation resultOp);
        object to_proto();
    }
}
