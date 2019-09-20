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
using Tensorflow.Queues;

namespace Tensorflow
{
    public partial class tensorflow
    {
        /// <summary>
        /// A FIFOQueue that supports batching variable-sized tensors by padding.
        /// </summary>
        /// <param name="capacity"></param>
        /// <param name="dtypes"></param>
        /// <param name="shapes"></param>
        /// <param name="names"></param>
        /// <param name="shared_name"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public PaddingFIFOQueue PaddingFIFOQueue(int capacity,
            TF_DataType[] dtypes,
            TensorShape[] shapes,
            string[] names = null,
            string shared_name = null,
            string name = "padding_fifo_queue")
            => new PaddingFIFOQueue(capacity,
                dtypes,
                shapes,
                names,
                shared_name: shared_name,
                name: name);
    }
}
