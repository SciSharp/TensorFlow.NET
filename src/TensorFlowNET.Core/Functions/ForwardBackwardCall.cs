using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Functions
{
    /// <summary>
    /// Holds the state of a function call between execution and recording.
    /// </summary>
    public class ForwardBackwardCall
    {
        TapeGradientFunctions _functions;
        Tensors _inference_args;
        Tensors _input_tangents;
        bool _tape_watching;
        EagerDefinedFunction forward_function;

        public ForwardBackwardCall(TapeGradientFunctions functions, 
            Tensors inference_args, 
            bool tape_watching)
        {
            _functions = functions;
            _inference_args = inference_args;
            _tape_watching = tape_watching;
        }
        
        public (EagerDefinedFunction, Tensors) Forward()
        {
            if (forward_function == null)
                forward_function = _functions.Forward(_inference_args);
            return (forward_function, _inference_args);
        }

        public void Record(Tensors flat_outputs)
        {
            if (_tape_watching && flat_outputs != null)
                _functions.Record(flat_outputs, _inference_args);
        }
    }
}
