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
using System.Collections.Generic;

namespace Tensorflow
{
    public partial class ops
    {
        _DefaultStack _default_session_stack = new _DefaultStack();

        public class _DefaultStack : ITensorFlowObject
        {
            Stack<object> stack;
#pragma warning disable CS0414 // The field 'ops._DefaultStack._enforce_nesting' is assigned but its value is never used
            bool _enforce_nesting = true;
#pragma warning restore CS0414 // The field 'ops._DefaultStack._enforce_nesting' is assigned but its value is never used

            public _DefaultStack()
            {
                stack = new Stack<object>();
            }

            public void __enter__()
            {
                
            }

            public void __exit__()
            {
                
            }

            public void Dispose()
            {
                throw new NotImplementedException();
            }

            public void __init__()
            {
                
            }

            public void __del__()
            {
                
            }
        }
    }
}
