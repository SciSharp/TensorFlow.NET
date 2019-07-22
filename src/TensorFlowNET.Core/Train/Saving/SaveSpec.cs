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

namespace Tensorflow
{
    /// <summary>
    /// Class used to describe tensor slices that need to be saved.
    /// </summary>
    public class SaveSpec
    {
        private Tensor _tensor;
        public Tensor tensor => _tensor;

        private string _slice_spec;
        public string slice_spec => _slice_spec;

        private string _name;
        public string name => _name;

        private TF_DataType _dtype;
        public TF_DataType dtype => _dtype;

        public SaveSpec(Tensor tensor, string slice_spec, string name, TF_DataType dtype = TF_DataType.DtInvalid)
        {
            _tensor = tensor;
            _slice_spec = slice_spec;
            _name = name;
            _dtype = dtype;
        }
    }
}
