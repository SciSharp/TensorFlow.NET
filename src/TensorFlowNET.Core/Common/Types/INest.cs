using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Common.Types
{
    /// <summary>
    /// This interface indicates that a class may have a nested structure and provide
    /// methods to manipulate with the structure.
    /// </summary>
    public interface INestStructure<T>: INestable<T>
    {
        /// <summary>
        /// Flatten the Nestable object. Node that if the object contains only one value, 
        /// it will be flattened to an enumerable with one element.
        /// </summary>
        /// <returns></returns>
        IEnumerable<T> Flatten();
        /// <summary>
        /// Construct a new object with the same nested structure.
        /// </summary>
        /// <typeparam name="TOut"></typeparam>
        /// <param name="func"></param>
        /// <returns></returns>
        INestStructure<TOut> MapStructure<TOut>(Func<T, TOut> func);
    }
}
