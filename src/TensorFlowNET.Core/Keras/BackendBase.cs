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
using static Tensorflow.Python;

namespace Tensorflow.Keras
{
    public abstract class BackendBase
    {
        TF_DataType _FLOATX = dtypes.float32;
        float _EPSILON = 1e-7f;
        ImageDataFormat _IMAGE_DATA_FORMAT = ImageDataFormat.channels_last;


        public float epsilon() => _EPSILON;

        public void set_epsilon(float e) => _EPSILON = e;

        public TF_DataType floatx() => _FLOATX;

        public void set_floatx(TF_DataType floatx) => _FLOATX = floatx;

        //public NDArray cast_to_floatx(NDArray x) => np.array(x, dtype: _FLOATX.as_numpy_datatype());

        public ImageDataFormat image_data_format() => _IMAGE_DATA_FORMAT;

        public void set_image_data_format(ImageDataFormat data_format) => _IMAGE_DATA_FORMAT = data_format;

        public ImageDataFormat normalize_data_format(object value = null)
        {
            if (value == null)
                value = _IMAGE_DATA_FORMAT;
            if (isinstance(value, typeof(ImageDataFormat)))
                return (ImageDataFormat)value;
            else if (isinstance(value, typeof(string)))
            {
                ImageDataFormat dataFormat;
                if(Enum.TryParse((string)value, true, out dataFormat))
                {
                    if (Enum.IsDefined(typeof(ImageDataFormat), dataFormat) | dataFormat.ToString().Contains(","))
                        return dataFormat;
                }
            }
            throw new Exception("The `data_format` argument must be one of \"channels_first\", \"channels_last\". Received: " + value.ToString());
        }

        //Legacy Methods

        public void set_image_dim_ordering(ImageDimOrder dim_ordering)
        {
            if (dim_ordering == ImageDimOrder.th)
                _IMAGE_DATA_FORMAT = ImageDataFormat.channels_first;
            else if (dim_ordering == ImageDimOrder.tf)
                _IMAGE_DATA_FORMAT = ImageDataFormat.channels_last;
            else
                throw new Exception("Unknown dim_ordering:"+ dim_ordering);
        }

        public ImageDimOrder image_dim_ordering()
        {
            if (_IMAGE_DATA_FORMAT == ImageDataFormat.channels_first)
                return ImageDimOrder.th;
            else
                return ImageDimOrder.tf;
        }
    }
    public enum ImageDimOrder
    {
        tf,
        th
    }
}
