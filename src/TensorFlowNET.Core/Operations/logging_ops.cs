﻿/*****************************************************************************
   Copyright 2021 Haiping Chen. All Rights Reserved.

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

using Tensorflow.Contexts;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class logging_ops
    {
        public Tensor print_v2(Tensor input, string output_stream = "stderr", string end = "\n", string name = null)
        {
            var formatted_string = tf.strings.format("{}",
                    new[] { input },
                    placeholder: "{}",
                    summarize: 3,
                    name: name);

            return tf.Context.ExecuteOp("PrintV2", name, new ExecuteOpArgs(formatted_string)
               .SetAttributes(new { output_stream, end })).SingleOrNull;
        }
    }
}
