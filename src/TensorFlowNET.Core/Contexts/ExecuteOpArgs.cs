using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class ExecuteOpArgs
    {
        public Func<Operation, object> GetGradientAttrs { get; set; }
        public object[] OpInputArgs { get; set; }
        public Dictionary<string, object> OpAttrs { get; set; }
        
        public ExecuteOpArgs(params object[] inputArgs)
        {
            OpInputArgs = inputArgs;
        }

        public ExecuteOpArgs SetAttributes(object attrs)
        {
            OpAttrs = ConvertToDict(attrs);
            return this;
        }
    }
}
