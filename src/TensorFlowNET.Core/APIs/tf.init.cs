using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations.Initializers;

namespace Tensorflow
{
    public static partial class tf
    {
        public static IInitializer zeros_initializer => new Zeros();
        public static IInitializer ones_initializer => new Ones();
        public static IInitializer glorot_uniform_initializer => new GlorotUniform();
        
        public static variable_scope variable_scope(string name,
               string default_name = null,
               object values = null,
               bool auxiliary_name_scope = true) => new variable_scope(name, 
                   default_name, 
                   values,
                   auxiliary_name_scope);

        public static variable_scope variable_scope(VariableScope scope,
              string default_name = null,
              object values = null,
              bool? reuse = null,
              bool auxiliary_name_scope = true) => new variable_scope(scope,
                  default_name,
                  values,
                  auxiliary_name_scope);
    }
}
