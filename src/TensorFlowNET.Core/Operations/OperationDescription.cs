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

using System;

namespace Tensorflow
{
    public class OperationDescription
    {
        private IntPtr _handle;
        public IntPtr op => _handle;

        public OperationDescription(Graph graph, string opType, string opName)
        {
            _handle = c_api.TF_NewOperation(graph, opType, opName);
        }

        public OperationDescription(IntPtr handle)
        {
            _handle = handle;
        }

        public void AddInputList(params TF_Output[] inputs)
        {
            c_api.TF_AddInputList(_handle, inputs, inputs.Length);
        }

        public void SetAttrType(string attr_name, TF_DataType value)
        {
            c_api.TF_SetAttrType(_handle, attr_name, value);
        }

        public void SetAttrShape(string attr_name, long[] dims)
        {
            c_api.TF_SetAttrShape(_handle, attr_name, dims, dims.Length);
        }

        public Operation FinishOperation(Status status)
        {
            return c_api.TF_FinishOperation(_handle, status);
        }

        public static implicit operator OperationDescription(IntPtr handle)
        {
            return new OperationDescription(handle);
        }

        public static implicit operator IntPtr(OperationDescription desc)
        {
            return desc._handle;
        }
    }
}
