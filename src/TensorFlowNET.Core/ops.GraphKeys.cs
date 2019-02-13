using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public partial class ops
    {
        /// <summary>
        /// Standard names to use for graph collections.
        /// The standard library uses various well-known names to collect and
        /// retrieve values associated with a graph. For example, the
        /// `tf.Optimizer` subclasses default to optimizing the variables
        /// collected under `tf.GraphKeys.TRAINABLE_VARIABLES` if none is
        /// specified, but it is also possible to pass an explicit list of
        /// variables.
        /// </summary>
        public static class GraphKeys
        {
            /// <summary>
            /// the subset of `Variable` objects that will be trained by an optimizer.
            /// </summary>
            public static string TRAINABLE_VARIABLES = "trainable_variables";

            /// <summary>
            /// Key to collect Variable objects that are global (shared across machines).
            /// Default collection for all variables, except local ones.
            /// </summary>
            public static string GLOBAL_VARIABLES = "variables";

            public static string[] _VARIABLE_COLLECTIONS = new string[] { "trainable_variables" }; 
            /// <summary>
            /// Key to collect BaseSaverBuilder.SaveableObject instances for checkpointing.
            /// </summary>
            public static string SAVEABLE_OBJECTS = "saveable_objects";
        }
    }
}
