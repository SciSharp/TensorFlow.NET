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

using static Tensorflow.Binding;

namespace Tensorflow
{
    public class gen_ctc_ops
    {
        public static Tensor[] ctc_greedy_decoder(Tensor inputs, Tensor sequence_length, bool merge_repeated = true, string name = "CTCGreedyDecoder")
        {
            var op = tf._op_def_lib._apply_op_helper("CTCGreedyDecoder", name: name, args: new
            {
                inputs,
                sequence_length,
                merge_repeated
            });
            /*var decoded_indices = op.outputs[0];
            var decoded_values = op.outputs[1];
            var decoded_shape = op.outputs[2];
            var log_probability = op.outputs[3];*/
            return op.outputs;
        }
    }
}
