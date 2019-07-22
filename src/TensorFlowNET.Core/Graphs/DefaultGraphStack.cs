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

using System.Collections.Generic;
using System.Linq;

namespace Tensorflow
{
    public class DefaultGraphStack
    {
        List<StackModel> stack = new List<StackModel>();

        public void set_controller(Graph @default)
        {
            if (!stack.Exists(x => x.Graph == @default))
                stack.Add(new StackModel { Graph = @default, IsDefault = true });

            foreach (var s in stack)
                s.IsDefault = s.Graph == @default;
        }

        public Graph get_controller()
        {
            if (stack.Count == 0)
                stack.Add(new StackModel { Graph = tf.Graph(), IsDefault = true });

            return stack.First(x => x.IsDefault).Graph;
        }

        public void reset()
        {
            stack.Clear();
        }
    }

    public class StackModel
    {
        public Graph Graph { get; set; }
        public bool IsDefault { get; set; }
    }
}
