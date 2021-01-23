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

using System.Collections.Generic;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class gen_logging_ops
    {
        public static Operation assert(Tensor condition, object[] data, long summarize = 3, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "Assert", name,
                    null,
                    new object[] { condition, data, summarize });

                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("Assert", name, args: new { condition, data, summarize });

            return _op;
        }

        public static Tensor histogram_summary(string tag, Tensor values, string name = null)
        {
            var dict = new Dictionary<string, object>();
            var op = tf.OpDefLib._apply_op_helper("HistogramSummary", name: name, args: new { tag, values });
            return op.output;
        }

        /// <summary>
        ///    Outputs a <c>Summary</c> protocol buffer with scalar values.
        /// </summary>
        /// <param name="tags">
        ///    Tags for the summary.
        /// </param>
        /// <param name="values">
        ///    Same shape as <c>tags</c>.  Values for the summary.
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'ScalarSummary'.
        /// </param>
        /// <returns>
        ///    Scalar.  Serialized <c>Summary</c> protocol buffer.
        ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        /// <remarks>
        ///    The input <c>tags</c> and <c>values</c> must have the same shape.  The generated summary
        ///    has a summary value for each tag-value pair in <c>tags</c> and <c>values</c>.
        /// </remarks>
        public static Tensor scalar_summary(string tags, Tensor values, string name = "ScalarSummary")
        {
            var dict = new Dictionary<string, object>();
            dict["tags"] = tags;
            dict["values"] = values;
            var op = tf.OpDefLib._apply_op_helper("ScalarSummary", name: name, keywords: dict);
            return op.output;
        }

        /// <summary>
        ///    Merges summaries.
        /// </summary>
        /// <param name="inputs">
        ///    Can be of any shape.  Each must contain serialized <c>Summary</c> protocol
        ///    buffers.
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'MergeSummary'.
        /// </param>
        /// <returns>
        ///    Scalar. Serialized <c>Summary</c> protocol buffer.
        ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        /// <remarks>
        ///    This op creates a
        ///    [<c>Summary</c>](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
        ///    protocol buffer that contains the union of all the values in the input
        ///    summaries.
        ///    
        ///    When the Op is run, it reports an <c>InvalidArgument</c> error if multiple values
        ///    in the summaries to merge use the same tag.
        /// </remarks>
        public static Tensor merge_summary(Tensor[] inputs, string name = "MergeSummary")
        {
            var dict = new Dictionary<string, object>();
            dict["inputs"] = inputs;
            var op = tf.OpDefLib._apply_op_helper("MergeSummary", name: name, keywords: dict);
            return op.output;
        }
    }
}
