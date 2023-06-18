using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Common.Types
{
    public static class Nest
    {
        /// <summary>
        /// Pack the flat items to a nested sequence by the template.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="template"></param>
        /// <param name="flatItems"></param>
        /// <returns></returns>
        public static Nest<TOut> PackSequenceAs<T, TOut>(INestable<T> template, TOut[] flatItems)
        {
            return template.AsNest().PackSequence(flatItems);
        }

        /// <summary>
        /// Pack the flat items to a nested sequence by the template.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="template"></param>
        /// <param name="flatItems"></param>
        /// <returns></returns>
        public static Nest<T> PackSequenceAs<T>(INestable<T> template, List<T> flatItems)
        {
            return template.AsNest().PackSequence(flatItems.ToArray());
        }

        /// <summary>
        /// Flatten the nested object.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="nestedObject"></param>
        /// <returns></returns>
        public static IEnumerable<T> Flatten<T>(INestable<T> nestedObject)
        {
            return nestedObject.AsNest().Flatten();
        }

        /// <summary>
        /// Map the structure with specified function.
        /// </summary>
        /// <typeparam name="TIn"></typeparam>
        /// <typeparam name="TOut"></typeparam>
        /// <param name="func"></param>
        /// <param name="nestedObject"></param>
        /// <returns></returns>
        public static INestStructure<TOut> MapStructure<TIn, TOut>(Func<TIn, TOut> func, INestable<TIn> nestedObject)
        {
            return nestedObject.AsNest().MapStructure(func);
        }

        public static bool IsNested<T>(INestable<T> obj)
        {
            return obj.AsNest().IsNested();
        }
    }
}
