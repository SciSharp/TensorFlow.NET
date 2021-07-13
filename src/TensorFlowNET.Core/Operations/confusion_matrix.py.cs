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

using System;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class confusion_matrix
    {
        /// <summary>
        /// Squeeze last dim if ranks differ from expected by exactly 1.
        /// </summary>
        /// <param name="labels"></param>
        /// <param name="predictions"></param>
        /// <param name="expected_rank_diff"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static (Tensor, Tensor) remove_squeezable_dimensions(Tensor labels,
            Tensor predictions,
            int expected_rank_diff = 0,
            string name = null)
        {
            return tf_with(ops.name_scope(name, default_name: "remove_squeezable_dimensions", (labels, predictions)), delegate
            {
                predictions = ops.convert_to_tensor(predictions);
                labels = ops.convert_to_tensor(labels);
                var predictions_shape = predictions.shape;
                var predictions_rank = predictions_shape.ndim;
                var labels_shape = labels.shape;
                var labels_rank = labels_shape.ndim;
                if (labels_rank > -1 && predictions_rank > -1)
                {
                    // Use static rank.
                    var rank_diff = predictions_rank - labels_rank;
                    if (rank_diff == expected_rank_diff + 1)
                        predictions = array_ops.squeeze(predictions, new int[] { -1 });
                    else if (rank_diff == expected_rank_diff - 1)
                        labels = array_ops.squeeze(labels, new int[] { -1 });
                    return (labels, predictions);
                }

                // Use dynamic rank.
                throw new NotImplementedException("remove_squeezable_dimensions dynamic rank");
            });
        }
    }
}
