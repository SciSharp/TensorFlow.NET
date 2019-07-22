﻿/*****************************************************************************
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
    public class SaveableObject
    {
        public Tensor op;
        public SaveSpec[] specs;
        public string name;
        public string device;

        public SaveableObject()
        {

        }

        public SaveableObject(Tensor var, string slice_spec, string name)
        {

        }

        public SaveableObject(Tensor op, SaveSpec[] specs, string name)
        {
            this.op = op;
            this.specs = specs;
            this.name = name;
        }

        public virtual ITensorOrOperation restore(Tensor[] restored_tensors, TensorShape[] restored_shapes = null)
        {
            var restored_tensor = restored_tensors[0];
            return gen_state_ops.assign(op,
                restored_tensor,
                validate_shape: restored_shapes == null && tensor_util.to_shape(op.shape).is_fully_defined());
        }
    }
}
