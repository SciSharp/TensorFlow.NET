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

using Tensorflow.Exceptions;

namespace Tensorflow
{
    /// <summary>
    /// Class used to describe tensor slices that need to be saved.
    /// </summary>
    public class SaveSpec
    {
        private Tensor _tensor = null;
        private Func<Tensor> _tensor_creator = null;
        public Tensor tensor
        {
            get
            {
                if(_tensor is not null || _tensor_creator is null)
                {
                    return _tensor;
                }
                else
                {
                    return _tensor_creator();
                }
            }
        }

        internal Func<Tensor> TensorCreator => _tensor_creator;

        private string _slice_spec;
        public string slice_spec => _slice_spec;

        private string _name;
        public string name { get => _name; set => _name = value; }

        private TF_DataType _dtype;
        public TF_DataType dtype => _dtype;
        private string _device;
        public string device => _device;

        public SaveSpec(Tensor tensor, string slice_spec, string name, TF_DataType dtype = TF_DataType.DtInvalid, string device = null)
        {
            _tensor = tensor;
            _slice_spec = slice_spec;
            _name = name;
            _dtype = dtype;
            if(device is not null)
            {
                _device = device;
            }
            else
            {
                _device = tensor.Device;
            }
        }

        public SaveSpec(Func<Tensor> tensor_creator, string slice_spec, string name, TF_DataType dtype = TF_DataType.DtInvalid, string device = null)
        {
            _tensor_creator = tensor_creator;
            _slice_spec = slice_spec;
            _name = name;
            if(dtype == TF_DataType.DtInvalid || device is null)
            {
                throw new AssertionError("When passing a callable `tensor` to a SaveSpec, an explicit dtype and device must be provided.");
            }
            _dtype = dtype;
            _device = device;
        }
    }
}
