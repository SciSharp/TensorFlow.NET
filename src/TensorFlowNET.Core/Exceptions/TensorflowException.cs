using System;
using System.Runtime.Serialization;

namespace Tensorflow
{

    /// <summary>
    ///     Serves as a base class to all exceptions of Tensorflow.NET.
    /// </summary>
    [Serializable]
    public class TensorflowException : Exception
    {
        /// <summary>Initializes a new instance of the <see cref="T:System.Exception"></see> class.</summary>
        public TensorflowException()
        { }

        /// <summary>Initializes a new instance of the <see cref="T:System.Exception"></see> class with serialized data.</summary>
        /// <param name="info">The <see cref="T:System.Runtime.Serialization.SerializationInfo"></see> that holds the serialized object data about the exception being thrown.</param>
        /// <param name="context">The <see cref="T:System.Runtime.Serialization.StreamingContext"></see> that contains contextual information about the source or destination.</param>
        /// <exception cref="T:System.ArgumentNullException">The <paramref name="info">info</paramref> parameter is null.</exception>
        /// <exception cref="T:System.Runtime.Serialization.SerializationException">The class name is null or <see cref="P:System.Exception.HResult"></see> is zero (0).</exception>
        protected TensorflowException(SerializationInfo info, StreamingContext context) : base(info, context)
        { }

        /// <summary>Initializes a new instance of the <see cref="T:System.Exception"></see> class with a specified error message.</summary>
        /// <param name="message">The message that describes the error.</param>
        public TensorflowException(string message) : base(message)
        { }

        /// <summary>Initializes a new instance of the <see cref="T:System.Exception"></see> class with a specified error message and a reference to the inner exception that is the cause of this exception.</summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        /// <param name="innerException">The exception that is the cause of the current exception, or a null reference (Nothing in Visual Basic) if no inner exception is specified.</param>
        public TensorflowException(string message, Exception innerException) : base(message, innerException)
        { }
    }
}