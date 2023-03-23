using System;
using Tensorflow.Functions;
using Tensorflow.Train;

namespace Tensorflow
{
    public class Function: Trackable
    {
#pragma warning disable CS0169 // The field 'Function._handle' is never used
        private IntPtr _handle;
#pragma warning restore CS0169 // The field 'Function._handle' is never used
        
        public string Name { get; set; }
        public Function()
        {

        }
        
        public Function(string name)
        {
            Name = name;
        }
    }
}
