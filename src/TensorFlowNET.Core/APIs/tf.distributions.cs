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
        public distributions_internal distributions { get; } = new distributions_internal();

        public class distributions_internal
        {
            public Normal Normal(Tensor loc,
                Tensor scale,
                bool validate_args = false,
                bool allow_nan_stats = true,
                string name = "Normal") => new Normal(loc, scale, validate_args = false, allow_nan_stats = true, "Normal");
        }
    }
}
