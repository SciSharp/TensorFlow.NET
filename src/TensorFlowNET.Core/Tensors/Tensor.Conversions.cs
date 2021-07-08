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

using Tensorflow.Numpy;
using System;
using System.Diagnostics.CodeAnalysis;
using System.Text;
using Tensorflow.Framework.Models;
using static Tensorflow.Binding;

namespace Tensorflow
{
    [SuppressMessage("ReSharper", "InvokeAsExtensionMethod")]
    public partial class Tensor
    {
        public unsafe void CopyTo(NDArray nd)
        {
            //if (!nd.Shape.IsContiguous)
                //throw new ArgumentException("NDArray has to be contiguous (ndarray.Shape.IsContiguous).");

            var length = (int)(nd.size * nd.dtypesize);

            switch (nd.dtype)
            {
                /*case NumpyDType.Boolean:
                    {
                        CopyTo(new Span<bool>(nd.Address.ToPointer(), length));
                        break;
                    }
                case NumpyDType.Byte:
                    {
                        CopyTo(new Span<byte>(nd.Address.ToPointer(), length));
                        break;
                    }
                case NumpyDType.Int16:
                    {
                        CopyTo(new Span<short>(nd.Address.ToPointer(), length));
                        break;
                    }
                case NumpyDType.UInt16:
                    {
                        CopyTo(new Span<ushort>(nd.Address.ToPointer(), length));
                        break;
                    }
                case NumpyDType.Int32:
                    {
                        CopyTo(new Span<int>(nd.Address.ToPointer(), length));
                        break;
                    }
                case NumpyDType.UInt32:
                    {
                        CopyTo(new Span<uint>(nd.Address.ToPointer(), length));
                        break;
                    }
                case NumpyDType.Int64:
                    {
                        CopyTo(new Span<long>(nd.Address.ToPointer(), length));
                        break;
                    }
                case NumpyDType.UInt64:
                    {
                        CopyTo(new Span<ulong>(nd.Address.ToPointer(), length));
                        break;
                    }
                case NumpyDType.Char:
                    {
                        CopyTo(new Span<char>(nd.Address.ToPointer(), length));
                        break;
                    }
                case NumpyDType.Double:
                    {
                        CopyTo(new Span<double>(nd.Address.ToPointer(), length));
                        break;
                    }
                case NumpyDType.Single:
                    {
                        CopyTo(new Span<float>(nd.Address.ToPointer(), length));
                        break;
                    }*/
                default:
                    throw new NotSupportedException();
            }
        }

        public void CopyTo<T>(Span<T> destination) where T : unmanaged
        {
            throw new NotImplementedException("");
        }

        public TensorSpec ToTensorSpec()
            => new TensorSpec(shape, dtype, name);
    }
}