using System;
using System.Collections.Generic;
using System.Text;

namespace TensorFlowNET.Core
{
    public sealed class MonoPInvokeCallbackAttribute : Attribute
    {
        /// <summary>
        /// Use this constructor to annotate the type of the callback function that 
        /// will be invoked from unmanaged code.
        /// </summary>
        /// <param name="t">T.</param>
        public MonoPInvokeCallbackAttribute(Type t) { }
    }
}
