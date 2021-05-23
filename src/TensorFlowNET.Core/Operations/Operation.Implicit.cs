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

namespace Tensorflow
{
    /// <summary>
    /// Convert to other datatype implicitly
    /// </summary>
    public partial class Operation
    {
        // make sure the new op is in the same graph instance
        public static implicit operator Operation(IntPtr handle)
            => new Operation(handle);

        public static implicit operator IntPtr(Operation op)
            => op._handle;
        public static implicit operator Tensor(Operation op)
            => op.output;

        public override string ToString()
        {
            return _handle == IntPtr.Zero ? "tf.Operation Undefined" : $"<tf.Operation '{name}' type={OpType}>";
        }

        public override bool Equals(object obj)
        {
            switch (obj)
            {
                case IntPtr val:
                    return val == _handle;
                case Operation val:
                    return val._handle == _handle;
            }

            return base.Equals(obj);
        }
    }
}
