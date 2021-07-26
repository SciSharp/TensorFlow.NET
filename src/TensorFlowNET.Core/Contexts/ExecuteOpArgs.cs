using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class ExecuteOpArgs
    {
        public Func<Operation, object> GetGradientAttrs { get; set; }
        public object[] OpInputArgs { get; set; }
        public Dictionary<string, object> OpAttrs { get; set; }
        
        [DebuggerStepThrough]
        public ExecuteOpArgs(params object[] inputArgs)
        {
            OpInputArgs = inputArgs;
        }

        [DebuggerStepThrough]
        public ExecuteOpArgs SetAttributes(object attrs)
        {
            OpAttrs = ConvertToDict(attrs);
            return this;
        }
    }
}
