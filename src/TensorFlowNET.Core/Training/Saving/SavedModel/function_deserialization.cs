using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Functions;
using Tensorflow.Util;

namespace Tensorflow.Training.Saving.SavedModel
{
    public static class function_deserialization
    {
        /// <summary>
        /// Creates a `Function` from a `SavedFunction`.
        /// </summary>
        /// <param name="saved_concrete_function"></param>
        /// <param name="concrete_functions"></param>
        /// <returns></returns>
        public static ConcreteFunction recreate_function(SavedFunction saved_concrete_function,
            IDictionary<string, ConcreteFunction> concrete_functions)
        {
            var function_spec = _deserialize_function_spec_as_nonmethod(saved_concrete_function.FunctionSpec);
            return null;
        }

        public static ConcreteFunction setup_bare_concrete_function(SavedBareConcreteFunction saved_bare_concrete_function, 
            IDictionary<string, ConcreteFunction> concrete_functions)
        {
            var concrete_function = concrete_functions[saved_bare_concrete_function.ConcreteFunctionName];
            concrete_function.ArgKeywords = saved_bare_concrete_function.ArgumentKeywords.ToList();
            concrete_function.NumPositionArgs = saved_bare_concrete_function.AllowedPositionalArguments;

            var function_spec = _deserialize_function_spec_as_nonmethod(saved_bare_concrete_function.FunctionSpec);
            concrete_function.AddTograph();
            return concrete_function;
        }

        private static FunctionSpec _deserialize_function_spec_as_nonmethod(FunctionSpec function_spec_proto)
        {
            // TODO(Rinne)； revise the implementation.
            return new FunctionSpec()
            {
                Fullargspec = function_spec_proto.Fullargspec,
                IsMethod = function_spec_proto.IsMethod,
                InputSignature = function_spec_proto.InputSignature,
                JitCompile = function_spec_proto.JitCompile
            };
        }
    }
}
