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
        public class GraphKeys
        {
            #region const
            /// <summary>
            /// Key to collect concatenated sharded variables.
            /// </summary>
            public const string CONCATENATED_VARIABLES_ = "concatenated_variables";
            /// <summary>
            /// the subset of `Variable` objects that will be trained by an optimizer.
            /// </summary>
            public const string TRAINABLE_VARIABLES_ = "trainable_variables";

            /// <summary>
            /// Trainable resource-style variables.
            /// </summary>
            public const string TRAINABLE_RESOURCE_VARIABLES_ = "trainable_resource_variables";

            /// <summary>
            /// Key for streaming model ports.
            /// </summary>
            public const string _STREAMING_MODEL_PORTS_ = "streaming_model_ports";

            /// <summary>
            /// Key to collect losses
            /// </summary>
            public const string LOSSES_ = "losses";

            public const string LOCAL_VARIABLES_ = "local_variables";

            public const string METRIC_VARIABLES_ = "metric_variables";
            public const string MODEL_VARIABLES_ = "model_variables";

            public const string MOVING_AVERAGE_VARIABLES_ = "moving_average_variables";

            /// <summary>
            /// Key to collect Variable objects that are global (shared across machines).
            /// Default collection for all variables, except local ones.
            /// </summary>
            public const string GLOBAL_VARIABLES_ = "variables";

            public const string TRAIN_OP_ = "train_op";

            public const string GLOBAL_STEP_ = "global_step";

            /// <summary>
            /// List of all collections that keep track of variables.
            /// </summary>
            public string[] _VARIABLE_COLLECTIONS_ = new string[]
            {
                GLOBAL_VARIABLES_,
                LOCAL_VARIABLES_,
                METRIC_VARIABLES_,
                MODEL_VARIABLES_,
                TRAINABLE_VARIABLES_,
                MOVING_AVERAGE_VARIABLES_,
                CONCATENATED_VARIABLES_,
                TRAINABLE_RESOURCE_VARIABLES_
            };

            /// <summary>
            /// Key to collect BaseSaverBuilder.SaveableObject instances for checkpointing.
            /// </summary>
            public const string SAVEABLE_OBJECTS_ = "saveable_objects";
            /// <summary>
            /// Key to collect update_ops
            /// </summary>
            public const string UPDATE_OPS_ = "update_ops";

            // Key to collect summaries.
            public const string SUMMARIES_ = "summaries";

            // Used to store v2 summary names.
            public const string _SUMMARY_COLLECTION_ = "_SUMMARY_V2";

            // Key for control flow context.
            public const string COND_CONTEXT_ = "cond_context";
            public const string WHILE_CONTEXT_ = "while_context";

            #endregion


            public string CONCATENATED_VARIABLES => CONCATENATED_VARIABLES_;
            /// <summary>
            /// the subset of `Variable` objects that will be trained by an optimizer.
            /// </summary>
            public string TRAINABLE_VARIABLES => TRAINABLE_VARIABLES_;

            /// <summary>
            /// Trainable resource-style variables.
            /// </summary>
            public string TRAINABLE_RESOURCE_VARIABLES => TRAINABLE_RESOURCE_VARIABLES_;

            /// <summary>
            /// Key for streaming model ports.
            /// </summary>
            public string _STREAMING_MODEL_PORTS => _STREAMING_MODEL_PORTS_;

            /// <summary>
            /// Key to collect local variables that are local to the machine and are not
            /// saved/restored.
            /// </summary>
            public string LOCAL_VARIABLES = LOCAL_VARIABLES_;

            /// <summary>
            /// Key to collect losses
            /// </summary>
            public string LOSSES => LOSSES_;

            public string METRIC_VARIABLES => METRIC_VARIABLES_;
            public string MOVING_AVERAGE_VARIABLES = MOVING_AVERAGE_VARIABLES_;

            /// <summary>
            /// Key to collect Variable objects that are global (shared across machines).
            /// Default collection for all variables, except local ones.
            /// </summary>
            public string GLOBAL_VARIABLES => GLOBAL_VARIABLES_;

            public string TRAIN_OP => TRAIN_OP_;

            public string GLOBAL_STEP => GLOBAL_STEP_;
            public string GLOBAL_STEP_READ_KEY = "global_step_read_op_cache";

            public string[] _VARIABLE_COLLECTIONS => _VARIABLE_COLLECTIONS_;
            /// <summary>
            /// Key to collect BaseSaverBuilder.SaveableObject instances for checkpointing.
            /// </summary>
            public string SAVEABLE_OBJECTS => SAVEABLE_OBJECTS_;
            /// <summary>
            /// Key to collect update_ops
            /// </summary>
            public string UPDATE_OPS => UPDATE_OPS_;

            // Key to collect summaries.
            public string SUMMARIES => SUMMARIES_;

            // Used to store v2 summary names.
            public string _SUMMARY_COLLECTION => _SUMMARY_COLLECTION_;

            // Key for control flow context.
            public string COND_CONTEXT => COND_CONTEXT_;
            public string WHILE_CONTEXT => WHILE_CONTEXT_;
        }
    }
}
