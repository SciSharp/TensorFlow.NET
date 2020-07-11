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
    public class ReferenceVariableSaveable : MySaveableObject
    {
        private SaveSpec _spec;

        public ReferenceVariableSaveable(Tensor var, string slice_spec, string name)
        {
            _spec = new SaveSpec(var, slice_spec, name, dtype: var.dtype);
            op = var;
            specs = new SaveSpec[] { _spec };
            this.name = name;
        }
    }
}
