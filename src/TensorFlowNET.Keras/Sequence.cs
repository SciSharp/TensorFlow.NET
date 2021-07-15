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

using Tensorflow.NumPy;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Tensorflow.Keras
{
    public class Sequence
    {
        /// <summary>
        /// Pads sequences to the same length.
        /// https://keras.io/preprocessing/sequence/
        /// https://faroit.github.io/keras-docs/1.2.0/preprocessing/sequence/
        /// </summary>
        /// <param name="sequences">List of lists, where each element is a sequence.</param>
        /// <param name="maxlen">Int, maximum length of all sequences.</param>
        /// <param name="dtype">Type of the output sequences.</param>
        /// <param name="padding">String, 'pre' or 'post':</param>
        /// <param name="truncating">String, 'pre' or 'post'</param>
        /// <param name="value">Float or String, padding value.</param>
        /// <returns></returns>
        public NDArray pad_sequences(IEnumerable<int[]> sequences,
            int? maxlen = null,
            string dtype = "int32",
            string padding = "pre",
            string truncating = "pre",
            object value = null)
        {
            if (value != null) throw new NotImplementedException("padding with a specific value.");
            if (padding != "pre" && padding != "post") throw new InvalidArgumentError("padding must be 'pre' or 'post'.");
            if (truncating != "pre" && truncating != "post") throw new InvalidArgumentError("truncating must be 'pre' or 'post'.");

            var length = sequences.Select(s => s.Length);

            if (maxlen == null)
                maxlen = length.Max();

            if (value == null)
                value = 0f;

            var type = dtypes.tf_dtype_from_name(dtype);
            var nd = np.zeros((length.Count(), maxlen.Value), dtype: type);

            for (int i = 0; i < nd.dims[0]; i++)
            {
                var s = sequences.ElementAt(i);
                if (s.Length > maxlen.Value)
                {
                    s = (truncating == "pre") ? s.Skip(s.Length - maxlen.Value).ToArray() : s.Take(maxlen.Value).ToArray();
                }
                var sliceString = (padding == "pre") ? $"{i},{maxlen - s.Length}:" : $"{i},:{s.Length}";
                var slices = sliceString.Split(',').Select(x => new Slice(x)).ToArray();
                nd[slices] = np.array(s);
            }

            return nd;
        }
    }
}
