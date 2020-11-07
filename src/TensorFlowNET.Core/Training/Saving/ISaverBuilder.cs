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
    public interface ISaverBuilder
    {
        Operation save_op(Tensor filename_tensor, MySaveableObject[] saveables);

        Tensor[] bulk_restore(Tensor filename_tensor, MySaveableObject[] saveables, int preferred_shard, bool restore_sequentially);

        SaverDef _build_internal(IVariableV1[] names_to_saveables,
            bool reshape = false,
            bool sharded = false,
            int max_to_keep = 5,
            float keep_checkpoint_every_n_hours = 10000,
            string name = null,
            bool restore_sequentially = false,
            string filename = "model",
            bool build_save = true,
            bool build_restore = true);
    }
}
