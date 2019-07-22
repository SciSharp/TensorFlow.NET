/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

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
            /// Trainable resource-style variables.
            /// </summary>
            public static string TRAINABLE_RESOURCE_VARIABLES = "trainable_resource_variables";

            /// <summary>
            /// Key for streaming model ports.
            /// </summary>
            public static string _STREAMING_MODEL_PORTS = "streaming_model_ports";

            /// <summary>
            /// Key to collect losses
            /// </summary>
            public const string LOSSES = "losses";

            /// <summary>
            /// Key to collect Variable objects that are global (shared across machines).
            /// Default collection for all variables, except local ones.
            /// </summary>
            public static string GLOBAL_VARIABLES = "variables";

            public static string TRAIN_OP = "train_op";

            public static string GLOBAL_STEP = GLOBAL_STEP = "global_step";

            public static string[] _VARIABLE_COLLECTIONS = new string[] { "variables", "trainable_variables", "model_variables" }; 
            /// <summary>
            /// Key to collect BaseSaverBuilder.SaveableObject instances for checkpointing.
            /// </summary>
            public static string SAVEABLE_OBJECTS = "saveable_objects";
            /// <summary>
            /// Key to collect update_ops
            /// </summary>
            public static string UPDATE_OPS = "update_ops";

            // Key to collect summaries.
            public const string SUMMARIES = "summaries";

            // Used to store v2 summary names.
            public static string _SUMMARY_COLLECTION = "_SUMMARY_V2";

            // Key for control flow context.
            public static string COND_CONTEXT = "cond_context";
            public static string WHILE_CONTEXT = "while_context";
        }
    }
}
