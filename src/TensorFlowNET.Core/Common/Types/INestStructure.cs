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
        NestType NestType { get; }

        /// <summary>
        /// The item count of depth 1 of the nested structure.
        /// For example, [1, 2, [3, 4, 5]] has ShallowNestedCount = 3.
        /// </summary>
        int ShallowNestedCount { get; }
        /// <summary>
        /// The total item count of depth 1 of the nested structure.
        /// For example, [1, 2, [3, 4, 5]] has TotalNestedCount = 5.
        /// </summary>
        int TotalNestedCount { get; }

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
