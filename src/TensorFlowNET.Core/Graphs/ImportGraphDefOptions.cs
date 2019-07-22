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
    public class ImportGraphDefOptions : IDisposable
    {
        private IntPtr _handle;

        public int NumReturnOutputs => c_api.TF_ImportGraphDefOptionsNumReturnOutputs(_handle);

        public ImportGraphDefOptions()
        {
            _handle = c_api.TF_NewImportGraphDefOptions();
        }

        public ImportGraphDefOptions(IntPtr handle)
        {
            _handle = handle;
        }

        public void AddReturnOutput(string name, int index)
        {
            c_api.TF_ImportGraphDefOptionsAddReturnOutput(_handle, name, index);
        }

        public void Dispose()
        {
            c_api.TF_DeleteImportGraphDefOptions(_handle);
        }

        public static implicit operator IntPtr(ImportGraphDefOptions opts) => opts._handle;
        public static implicit operator ImportGraphDefOptions(IntPtr handle) => new ImportGraphDefOptions(handle);
    }
}
