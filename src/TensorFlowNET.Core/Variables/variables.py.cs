using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class variables
    {
        /// <summary>
        /// Returns all variables created with `trainable=True`
        /// </summary>
        /// <returns></returns>
        public static object trainable_variables()
        {
            return ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES);
        }
    }
}
