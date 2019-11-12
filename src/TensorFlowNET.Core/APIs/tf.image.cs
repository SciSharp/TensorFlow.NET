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

            /// <summary>
            /// Extracts crops from the input image tensor and resizes them using bilinear sampling or nearest neighbor sampling (possibly with aspect ratio change) to a common output size specified by crop_size. This is more general than the crop_to_bounding_box op which extracts a fixed size slice from the input image and does not allow resizing or aspect ratio change.
            /// Returns a tensor with crops from the input image at positions defined at the bounding box locations in boxes.The cropped boxes are all resized(with bilinear or nearest neighbor interpolation) to a fixed size = [crop_height, crop_width].The result is a 4 - D tensor[num_boxes, crop_height, crop_width, depth].The resizing is corner aligned. In particular, if boxes = [[0, 0, 1, 1]], the method will give identical results to using tf.image.resize_bilinear() or tf.image.resize_nearest_neighbor() (depends on the method argument) with align_corners = True.
            /// </summary>
            /// <param name="image">A Tensor. Must be one of the following types: uint8, uint16, int8, int16, int32, int64, half, float32, float64. A 4-D tensor of shape [batch, image_height, image_width, depth]. Both image_height and image_width need to be positive.</param>
            /// <param name="boxes">A Tensor of type float32. A 2-D tensor of shape [num_boxes, 4]. The i-th row of the tensor specifies the coordinates of a box in the box_ind[i] image and is specified in normalized coordinates [y1, x1, y2, x2]. A normalized coordinate value of y is mapped to the image coordinate at y * (image_height - 1), so as the [0, 1] interval of normalized image height is mapped to [0, image_height - 1] in image height coordinates. We do allow y1 > y2, in which case the sampled crop is an up-down flipped version of the original image. The width dimension is treated similarly. Normalized coordinates outside the [0, 1] range are allowed, in which case we use extrapolation_value to extrapolate the input image values.</param>
            /// <param name="box_ind">A Tensor of type int32. A 1-D tensor of shape [num_boxes] with int32 values in [0, batch). The value of box_ind[i] specifies the image that the i-th box refers to.</param>
            /// <param name="crop_size">A Tensor of type int32. A 1-D tensor of 2 elements, size = [crop_height, crop_width]. All cropped image patches are resized to this size. The aspect ratio of the image content is not preserved. Both crop_height and crop_width need to be positive.</param>
            /// <param name="method">An optional string from: "bilinear", "nearest". Defaults to "bilinear". A string specifying the sampling method for resizing. It can be either "bilinear" or "nearest" and default to "bilinear". Currently two sampling methods are supported: Bilinear and Nearest Neighbor.</param>
            /// <param name="extrapolation_value">An optional float. Defaults to 0. Value used for extrapolation, when applicable.</param>
            /// <param name="name">A name for the operation (optional).</param>
            /// <returns></returns>
            public Tensor crop_and_resize(Tensor image, Tensor boxes, Tensor box_ind, Tensor crop_size, string method = "bilinear", float extrapolation_value = 0f, string name = null) =>
                image_ops_impl.crop_and_resize(image, boxes, box_ind, crop_size, method, extrapolation_value, name);


            public Tensor resize_bilinear(Tensor images, Tensor size, bool align_corners = false, string name = null)
                => gen_image_ops.resize_bilinear(images, size, align_corners: align_corners, name: name);

            public Tensor resize_images(Tensor images, Tensor size, ResizeMethod method = ResizeMethod.BILINEAR,
                    bool align_corners = false, bool preserve_aspect_ratio = false, string name = null)
                => image_ops_impl.resize_images(images, size, method: method,
                    align_corners: align_corners, preserve_aspect_ratio: preserve_aspect_ratio, name: name);

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
