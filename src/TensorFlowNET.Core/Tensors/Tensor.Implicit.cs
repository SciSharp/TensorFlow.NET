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
    public partial class Tensor
    {
        /// <summary>
        /// Issue unresolved, will cause name_scope problem.
        /// </summary>
        /// <param name="scalar"></param>
        /*public static implicit operator Tensor(double scalar)
        {
            return constant_op.constant(scalar);
        }*/

        /*public static implicit operator Tensor(int scalar)
        {
            return constant_op.constant(scalar);
        }*/

        public static implicit operator int(Tensor tensor)
        {
            return tensor.Data<int>()[0];
        }

        public static implicit operator IntPtr(Tensor tensor)
        {
            if (tensor._handle == IntPtr.Zero)
                Console.WriteLine("tensor is not allocated.");
            return tensor._handle;
        }

        public static implicit operator Operation(Tensor tensor)
        {
            return tensor.op;
        }

        public static implicit operator Tensor(IntPtr handle)
        {
            return new Tensor(handle);
        }
    }
}
