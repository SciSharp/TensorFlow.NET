using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Common.Types
{
    /// <summary>
    /// This interface is used when some corresponding python methods have optional args.
    /// For example, `Keras.Layer.Apply` generally takes three args as the inputs, while 
    /// `Keras.Layer.RNN` takes more. Then when calling RNN, you should add `RnnOptionalArgs` 
    /// as the parameter of the method.
    /// </summary>
    public interface IOptionalArgs
    {
        /// <summary>
        /// The identifier of the class. It is not an argument but only something to 
        /// separate different OptionalArgs.
        /// </summary>
        string Identifier { get; }
    }
}
