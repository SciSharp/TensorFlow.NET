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

            public Tensor decode_image(Tensor contents, int channels = 0, TF_DataType dtype = TF_DataType.TF_UINT8,
                string name = null, bool expand_animations = true)
                => image_ops_impl.decode_image(contents, channels: channels, dtype: dtype,
                    name: name, expand_animations: expand_animations);

            /// <summary>
            /// Convenience function to check if the 'contents' encodes a JPEG image.
            /// </summary>
            /// <param name="contents"></param>
            /// <param name="name"></param>
            /// <returns></returns>
            public Tensor is_jpeg(Tensor contents, string name = null)
                => image_ops_impl.is_jpeg(contents, name: name);

            /// <summary>
            /// Resize `images` to `size` using nearest neighbor interpolation.
            /// </summary>
            /// <param name="images"></param>
            /// <param name="size"></param>
            /// <param name="align_corners"></param>
            /// <param name="name"></param>
            /// <param name="half_pixel_centers"></param>
            /// <returns></returns>
            public Tensor resize_nearest_neighbor<Tsize>(Tensor images, Tsize size, bool align_corners = false,
                string name = null, bool half_pixel_centers = false)
                => image_ops_impl.resize_nearest_neighbor(images, size, align_corners: align_corners,
                    name: name, half_pixel_centers: half_pixel_centers);
        }
    }
}
