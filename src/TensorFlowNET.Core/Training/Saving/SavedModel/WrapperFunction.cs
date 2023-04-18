using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Functions;

namespace Tensorflow.Training.Saving.SavedModel
{
    /// <summary>
    /// A class wraps a concrete function to handle different distributed contexts.
    /// </summary>
    internal class WrapperFunction: ConcreteFunction
    {
        public WrapperFunction(ConcreteFunction concrete_function): base(concrete_function.func_graph)
        {
            throw new NotImplementedException();
            //this.forward_backward = concrete_function.forward_backward;
            //this.Outputs = concrete_function.Outputs;
            //this.ReturnType = concrete_function.ReturnType;
            //this.OutputStructure = concrete_function.OutputStructure;
            //this.ArgKeywords = concrete_function.ArgKeywords;
        }
    }
}
