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
        public static implicit operator Tensor(bool value)
        {
            return tf.constant(value, TF_DataType.TF_BOOL);
        }

        public static implicit operator Tensor(sbyte value)
        {
            return tf.constant(value, TF_DataType.TF_INT8);
        }

        public static implicit operator Tensor(byte value)
        {
            return tf.constant(value, TF_DataType.TF_INT16);
        }

        public static implicit operator Tensor(ushort value)
        {
            return tf.constant(value, TF_DataType.TF_UINT16);
        }

        public static implicit operator Tensor(short value)
        {
            return tf.constant(value, TF_DataType.TF_INT16);
        }

        public static implicit operator Tensor(int value)
        {
            return tf.constant(value, TF_DataType.TF_INT32);
        }

        public static implicit operator Tensor(uint value)
        {
            return tf.constant(value, TF_DataType.TF_UINT32);
        }

        public static implicit operator Tensor(long value)
        {
            return tf.constant(value, TF_DataType.TF_INT64);
        }

        public static implicit operator Tensor(ulong value)
        {
            return tf.constant(value, TF_DataType.TF_UINT64);
        }

        public static implicit operator Tensor(float value)
        {
            return tf.constant(value, TF_DataType.TF_FLOAT);
        }

        public static implicit operator Tensor(double value)
        {
            return tf.constant(value, TF_DataType.TF_DOUBLE);
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
