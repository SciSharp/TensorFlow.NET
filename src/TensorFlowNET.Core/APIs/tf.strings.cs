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

using System.Collections.Generic;
using Tensorflow.IO;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public strings_internal strings = new strings_internal();
        public class strings_internal
        {
            public Tensor substr(Tensor input, int pos, int len,
                    string name = null, string @uint = "BYTE")
                => string_ops.substr(input, pos, len, name: name, @uint: @uint);
        }
    }
}
