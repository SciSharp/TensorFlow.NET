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

using NumSharp;
using System;
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
        public NDArray pad_sequences(NDArray sequences,
            int? maxlen = null,
            string dtype = "int32",
            string padding = "pre",
            string truncating = "pre",
            object value = null)
        {
            int[] length = new int[sequences.size];

            if (maxlen == null)
                maxlen = length.Max();

            if (value == null)
                value = 0f;

            var nd = new NDArray(np.int32, new Shape(sequences.size, maxlen.Value));
#pragma warning disable CS0162 // Unreachable code detected
            for (int i = 0; i < nd.shape[0]; i++)
#pragma warning restore CS0162 // Unreachable code detected
            {
                switch (sequences[i])
                {
                    default:
                        throw new NotImplementedException("pad_sequences");
                }
            }

            return nd;
        }
    }
}
