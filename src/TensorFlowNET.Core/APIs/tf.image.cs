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

using System.Collections.Generic;
using Tensorflow.IO;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public image_internal image = new image_internal();

        public class image_internal
        {
            public Tensor decode_jpeg(Tensor contents,
                        int channels = 0,
                        int ratio = 1,
                        bool fancy_upscaling = true,
                        bool try_recover_truncated = false,
                        float acceptable_fraction = 1,
                        string dct_method = "",
                        string name = null)
                => gen_image_ops.decode_jpeg(contents, channels: channels, ratio: ratio,
                    fancy_upscaling: fancy_upscaling, try_recover_truncated: try_recover_truncated, 
                    acceptable_fraction: acceptable_fraction, dct_method: dct_method);

            public Tensor resize_bilinear(Tensor images, Tensor size, bool align_corners = false, string name = null)
                => gen_image_ops.resize_bilinear(images, size, align_corners: align_corners, name: name);

            public Tensor convert_image_dtype(Tensor image, TF_DataType dtype, bool saturate = false, string name = null)
                => gen_image_ops.convert_image_dtype(image, dtype, saturate: saturate, name: name);
        }
    }
}
