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
    public partial class tensorflow
    {
        /// <summary>
        /// Interleave the values from the data tensors into a single tensor.
        /// </summary>
        /// <param name="indices"></param>
        /// <param name="data"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor dynamic_stitch(Tensor[] indices, Tensor[] data, string name = null)
            => gen_data_flow_ops.dynamic_stitch(indices, data, name: name);

        /// <summary>
        /// Partitions `data` into `num_partitions` tensors using indices from `partitions`.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="partitions"></param>
        /// <param name="num_partitions">The number of partitions to output.</param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor[] dynamic_partition(Tensor data, Tensor partitions, int num_partitions,
            string name = null)
            => gen_data_flow_ops.dynamic_partition(data, partitions, num_partitions, name: name);
    }
}
