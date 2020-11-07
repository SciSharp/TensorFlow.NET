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
    public class ctc_ops
    {
        /// <summary>
        ///    Performs greedy decoding on the logits given in inputs.
        /// </summary>
        /// <param name="inputs">
        ///    3-D, shape: <c>(max_time x batch_size x num_classes)</c>, the logits.
        /// </param>
        /// <param name="sequence_length">
        ///    A vector containing sequence lengths, size <c>(batch_size)</c>.
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'CTCGreedyDecoder'.
        /// </param>
        /// <param name="merge_repeated">
        ///    If True, merge repeated classes in output.
        /// </param>
        /// <returns>
        ///    Returns a tuple with multiple values, as follows:
        ///    decoded_indices : Indices matrix, size <c>(total_decoded_outputs x 2)</c>,
        ///    of a <c>SparseTensor&amp;lt;int64, 2&amp;gt;</c>.  The rows store: [batch, time].
        ///    decoded_values : Values vector, size: <c>(total_decoded_outputs)</c>,
        ///    of a <c>SparseTensor&amp;lt;int64, 2&amp;gt;</c>.  The vector stores the decoded classes.
        ///    decoded_shape : Shape vector, size <c>(2)</c>, of the decoded SparseTensor.
        ///    Values are: <c>[batch_size, max_decoded_length]</c>.
        ///    log_probability : Matrix, size <c>(batch_size x 1)</c>, containing sequence
        ///    log-probabilities.
        ///    The Operation can be fetched from any of the Tensorreturned in the tuple values, by fetching the Operation property.
        /// </returns>
        /// <remarks>
        ///    A note about the attribute merge_repeated: if enabled, when
        ///    consecutive logits' maximum indices are the same, only the first of
        ///    these is emitted.  Labeling the blank '*', the sequence "A B B * B B"
        ///    becomes "A B B" if merge_repeated = True and "A B B B B" if
        ///    merge_repeated = False.
        ///    
        ///    Regardless of the value of merge_repeated, if the maximum index of a given
        ///    time and batch corresponds to the blank, index <c>(num_classes - 1)</c>, no new
        ///    element is emitted.
        /// </remarks>
        public Tensor[] ctc_greedy_decoder(Tensor inputs, Tensor sequence_length, bool merge_repeated = true, string name = null)
                => gen_ctc_ops.ctc_greedy_decoder(inputs, sequence_length, merge_repeated: merge_repeated, name: name);
    }
}
