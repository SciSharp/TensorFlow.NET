using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Common.Types;

namespace Tensorflow.Common.Extensions
{
    public static class NestExtensions
    {
        public static Tensors ToTensors(this INestable<Tensor> tensors)
        {
            return new Tensors(tensors.AsNest());
        }

        public static Tensors? ToTensors(this Nest<Tensor> tensors)
        {
            return Tensors.FromNest(tensors);
        }

        /// <summary>
        /// If the nested object is already a nested type, this function could reduce it.
        /// For example, `Nest[Nest[T]]` can be reduced to `Nest[T]`.
        /// </summary>
        /// <typeparam name="TIn"></typeparam>
        /// <typeparam name="TOut"></typeparam>
        /// <param name="input"></param>
        /// <returns></returns>
        public static Nest<TOut> ReduceTo<TIn, TOut>(this INestStructure<TIn> input) where TIn: INestStructure<TOut>
        {
            return Nest<TOut>.ReduceFrom(input);
        }
    }
}
