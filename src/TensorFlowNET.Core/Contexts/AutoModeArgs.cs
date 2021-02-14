using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class AutoModeArgs
    {
        public Func<Operation, object> GetGradientAttrs { get; set; }
        public object OpInputArgs { get; set; }
        public object OpAttrs { get; set; }
    }
}
