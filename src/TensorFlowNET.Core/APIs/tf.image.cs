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

using OneOf.Types;
using System;
using System.Buffers.Text;
using Tensorflow.Contexts;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public image_internal image = new image_internal();

        public class image_internal
        {
            public Tensor random_flip_up_down(Tensor image, int seed = 0)
                => image_ops_impl.random_flip_up_down(image, seed);

            public Tensor random_flip_left_right(Tensor image, int seed = 0)
                => image_ops_impl.random_flip_left_right(image, seed);

            public Tensor flip_left_right(Tensor image)
                => image_ops_impl.flip_left_right(image);

            public Tensor flip_up_down(Tensor image)
                => image_ops_impl.flip_up_down(image);

            public Tensor rot90(Tensor image, int k = 1, string name = null)
                => image_ops_impl.rot90(image, k, name);

            public Tensor transpose(Tensor image, string name = null)
                => image_ops_impl.transpose(image, name);

            public Tensor central_crop(Tensor image, float central_fraction)
                => image_ops_impl.central_crop(image, central_fraction);

            public Tensor pad_to_bounding_box(Tensor image, int offset_height, int offset_width, int target_height, int target_width)
                => image_ops_impl.pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width);

            public Tensor crop_to_bounding_box(Tensor image, int offset_height, int offset_width, int target_height, int target_width)
                => image_ops_impl.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width);

            public Tensor resize_image_with_crop_or_pad(Tensor image, object target_height, object target_width)
                => image_ops_impl.resize_image_with_crop_or_pad(image, target_height, target_width);

            public Tensor resize_images(Tensor images, Tensor size, string method = ResizeMethod.BILINEAR, bool preserve_aspect_ratio = false, bool antialias = false,
                string name = null)
                => image_ops_impl.resize_images(images, size, method, preserve_aspect_ratio, antialias, name);

            public Tensor resize_images_v2(Tensor images, Shape size, string method = ResizeMethod.BILINEAR, bool preserve_aspect_ratio = false, bool antialias = false,
                string name = null)
                => image_ops_impl.resize_images_v2(images, size, method, preserve_aspect_ratio, antialias, name);

            public Tensor resize_images_v2(Tensor images, Tensor size, string method = ResizeMethod.BILINEAR, bool preserve_aspect_ratio = false, bool antialias = false,
                string name = null)
                => image_ops_impl.resize_images_v2(images, size, method, preserve_aspect_ratio, antialias, name);

            public Tensor resize_images_with_pad(Tensor image, int target_height, int target_width, string method, bool antialias)
                => image_ops_impl.resize_images_with_pad(image, target_height, target_width, method, antialias);

            public Tensor per_image_standardization(Tensor image)
                => image_ops_impl.per_image_standardization(image);

            public Tensor random_brightness(Tensor image, float max_delta, int seed = 0)
                => image_ops_impl.random_brightness(image, max_delta, seed);

            public Tensor random_contrast(Tensor image, float lower, float upper, int seed = 0)
                => image_ops_impl.random_contrast(image, lower, upper, seed);

            public Tensor adjust_brightness(Tensor image, Tensor delta)
                => image_ops_impl.adjust_brightness(image, delta);

            public Tensor adjust_contrast(Tensor images, Tensor contrast_factor)
                => image_ops_impl.adjust_contrast(images, contrast_factor);

            public Tensor adjust_gamma(Tensor image, int gamma = 1, int gain = 1)
                => image_ops_impl.adjust_gamma(image, gamma, gain);

            public Tensor rgb_to_grayscale(Tensor images, string name = null)
                => image_ops_impl.rgb_to_grayscale(images, name);

            public Tensor grayscale_to_rgb(Tensor images, string name = null)
                => image_ops_impl.grayscale_to_rgb(images, name);

            public Tensor random_hue(Tensor image, float max_delta, int seed = 0)
                => image_ops_impl.random_hue(image, max_delta, seed);

            public Tensor adjust_hue(Tensor image, Tensor delta, string name = null)
                => image_ops_impl.adjust_hue(image, delta, name);

            public Tensor random_jpeg_quality(Tensor image, float min_jpeg_quality, float max_jpeg_quality, int seed = 0)
                => image_ops_impl.random_jpeg_quality(image, min_jpeg_quality, max_jpeg_quality, seed);

            public Tensor adjust_jpeg_quality(Tensor image, Tensor jpeg_quality, string name = null)
                => image_ops_impl.adjust_jpeg_quality(image, jpeg_quality, name);

            public Tensor random_saturation(Tensor image, float lower, float upper, int seed = 0)
                => image_ops_impl.random_saturation(image, lower, upper, seed);

            public Tensor adjust_saturation(Tensor image, Tensor saturation_factor, string name = null)
                => image_ops_impl.adjust_saturation(image, saturation_factor, name);

            public Tensor total_variation(Tensor images, string name = null)
                => image_ops_impl.total_variation(images, name);

            public (Tensor, Tensor, Tensor) sample_distorted_bounding_box(Tensor image_size, Tensor bounding_boxes,
                        int seed = 0,
                        Tensor min_object_covered = null,
                        float[] aspect_ratio_range = null,
                        float[] area_range = null,
                        int max_attempts = 100,
                        bool use_image_if_no_bounding_boxes = false,
                        string name = null)
                => image_ops_impl.sample_distorted_bounding_box_v2(image_size, bounding_boxes, seed, min_object_covered, aspect_ratio_range,
                                                                area_range, max_attempts, use_image_if_no_bounding_boxes, name);

            public Tensor non_max_suppression(Tensor boxes, Tensor scores, Tensor max_output_size, float iou_threshold = 0.5f,
                float score_threshold = -1f / 0f, /*float soft_nms_sigma = 0.0f,*/ string name = null)
                => image_ops_impl.non_max_suppression(boxes, scores, max_output_size, iou_threshold, score_threshold, name);

            public Tensor non_max_suppression_with_overlaps(Tensor overlaps, Tensor scores, Tensor max_output_size,
                float overlap_threshold = 0.5f, float score_threshold = -1 / 0f, string name = null)
                => image_ops_impl.non_max_suppression_with_overlaps(overlaps, scores, max_output_size, overlap_threshold, score_threshold, name);

            public Tensor rgb_to_yiq(Tensor images)
                => image_ops_impl.rgb_to_yiq(images);

            public Tensor yiq_to_rgb(Tensor images)
                => image_ops_impl.yiq_to_rgb(images);

            public Tensor rgb_to_yuv(Tensor images)
                => image_ops_impl.rgb_to_yuv(images);

            public Tensor yuv_to_rgb(Tensor images)
                => image_ops_impl.yuv_to_rgb(images);

            public Tensor psnr(Tensor a, Tensor b, Tensor max_val, string name = null)
                => image_ops_impl.psnr(a, b, max_val, name);

            public Tensor ssim(Tensor img1, Tensor img2, float max_val = 1f, float filter_size = 11f, float filter_sigma = 1.5f,
                float k1 = 0.01f, float k2 = 0.03f)
                => image_ops_impl.ssim(img1, img2, max_val, filter_size, filter_sigma, k1, k2);

            public Tensor ssim_multiscale(Tensor img1, Tensor img2, float max_val, float[] power_factors = null, float filter_size = 11f,
                float filter_sigma = 1.5f, float k1 = 0.01f, float k2 = 0.03f)
                => image_ops_impl.ssim_multiscale(img1, img2, max_val, power_factors, filter_size, filter_sigma, k1, k2);

            public (Tensor, Tensor) image_gradients(Tensor image)
                => image_ops_impl.image_gradients(image);

            public Tensor sobel_edges(Tensor image)
                => image_ops_impl.sobel_edges(image);

            /// <summary>
            /// Adjust contrast of RGB or grayscale images.
            /// </summary>
            /// <param name="images">Images to adjust.  At least 3-D.</param>
            /// <param name="contrast_factor"></param>
            /// <param name="name">A float multiplier for adjusting contrast.</param>
            /// <returns>The contrast-adjusted image or images.</returns>
            public Tensor adjust_contrast(Tensor images, float contrast_factor, string name = null)
                => gen_image_ops.adjust_contrastv2(images, contrast_factor, name);

            /// <summary>
            /// Adjust hue of RGB images.
            /// </summary>
            /// <param name="images">RGB image or images. The size of the last dimension must be 3.</param>
            /// <param name="delta">float.  How much to add to the hue channel.</param>
            /// <param name="name">A name for this operation (optional).</param>
            /// <returns>Adjusted image(s), same shape and DType as `image`.</returns>
            /// <exception cref="ValueError">if `delta` is not in the interval of `[-1, 1]`.</exception>
            public Tensor adjust_hue(Tensor images, float delta, string name = null)
            {
                if (tf.Context.executing_eagerly())
                {
                    if (delta < -1f || delta > 1f)
                        throw new ValueError("delta must be in the interval [-1, 1]");
                }
                return gen_image_ops.adjust_hue(images, delta, name: name);
            }

            /// <summary>
            /// Adjust saturation of RGB images.
            /// </summary>
            /// <param name="image">RGB image or images. The size of the last dimension must be 3.</param>
            /// <param name="saturation_factor">float. Factor to multiply the saturation by.</param>
            /// <param name="name">A name for this operation (optional).</param>
            /// <returns>Adjusted image(s), same shape and DType as `image`.</returns>
            public Tensor adjust_saturation(Tensor image, float saturation_factor, string name = null)
                => gen_image_ops.adjust_saturation(image, saturation_factor, name);

            /// <summary>
            /// Greedily selects a subset of bounding boxes in descending order of score.
            /// </summary>
            /// <param name="boxes">
            /// A 4-D float `Tensor` of shape `[batch_size, num_boxes, q, 4]`. If `q` 
            /// is 1 then same boxes are used for all classes otherwise, if `q` is equal 
            /// to number of classes, class-specific boxes are used.
            /// </param>
            /// <param name="scores">
            /// A 3-D float `Tensor` of shape `[batch_size, num_boxes, num_classes]` 
            /// representing a single score corresponding to each box(each row of boxes).
            /// </param>
            /// <param name="max_output_size_per_class">
            /// A scalar integer `Tensor` representing the 
            /// maximum number of boxes to be selected by non-max suppression per class
            /// </param>
            /// <param name="max_total_size">
            /// A int32 scalar representing maximum number of boxes retained 
            /// over all classes.Note that setting this value to a large number may 
            /// result in OOM error depending on the system workload.
            /// </param>
            /// <param name="iou_threshold">
            /// A float representing the threshold for deciding whether boxes 
            /// overlap too much with respect to IOU.
            /// </param>
            /// <param name="score_threshold">
            /// A float representing the threshold for deciding when to 
            /// remove boxes based on score.
            /// </param>
            /// <param name="pad_per_class">
            /// If false, the output nmsed boxes, scores and classes are 
            /// padded/clipped to `max_total_size`. If true, the output nmsed boxes, scores and classes are padded to be of length `max_size_per_class`*`num_classes`, 
            /// unless it exceeds `max_total_size` in which case it is clipped to `max_total_size`. Defaults to false.
            /// </param>
            /// <param name="clip_boxes">
            /// If true, the coordinates of output nmsed boxes will be clipped 
            /// to[0, 1]. If false, output the box coordinates as it is. Defaults to true.
            /// </param>
            /// <returns>
            /// 'nmsed_boxes': A [batch_size, max_detections, 4] float32 tensor containing the non-max suppressed boxes.
            /// 'nmsed_scores': A [batch_size, max_detections] float32 tensor containing the scores for the boxes.
            /// 'nmsed_classes': A [batch_size, max_detections] float32 tensor containing the class for boxes.
            /// 'valid_detections': A [batch_size] int32 tensor indicating the number of
            ///     valid detections per batch item. Only the top valid_detections[i] entries
            ///     in nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The rest of the
            ///     entries are zero paddings.
            /// </returns>
            public (Tensor, Tensor, Tensor, Tensor) combined_non_max_suppression(
                        Tensor boxes, 
                        Tensor scores, 
                        int max_output_size_per_class, 
                        int max_total_size,
                        float iou_threshold,
                        float score_threshold,
                        bool pad_per_class = false,
                        bool clip_boxes = true)
            {
                var iou_threshold_t = ops.convert_to_tensor(iou_threshold, TF_DataType.TF_FLOAT, name: "iou_threshold");
                var score_threshold_t = ops.convert_to_tensor(score_threshold, TF_DataType.TF_FLOAT, name: "score_threshold");
                var max_total_size_t = ops.convert_to_tensor(max_total_size);
                var max_output_size_per_class_t = ops.convert_to_tensor(max_output_size_per_class);
                return gen_image_ops.combined_non_max_suppression(boxes, scores, max_output_size_per_class_t, max_total_size_t,
                                       iou_threshold_t, score_threshold_t, pad_per_class, clip_boxes);
            }

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
            /// <returns>A 4-D tensor of shape [num_boxes, crop_height, crop_width, depth].</returns>
            public Tensor crop_and_resize(Tensor image, Tensor boxes, Tensor box_ind, Tensor crop_size, string method = "bilinear", float extrapolation_value = 0f, string name = null) =>
                gen_image_ops.crop_and_resize(image, boxes, box_ind, crop_size, method, extrapolation_value, name);

            public Tensor decode_jpeg(Tensor contents,
                        int channels = 0,
                        int ratio = 1,
                        bool fancy_upscaling = true,
                        bool try_recover_truncated = false,
                        int acceptable_fraction = 1,
                        string dct_method = "",
                        string name = null)
                => gen_image_ops.decode_jpeg(contents, channels: channels, ratio: ratio,
                    fancy_upscaling: fancy_upscaling, try_recover_truncated: try_recover_truncated,
                    acceptable_fraction: acceptable_fraction, dct_method: dct_method);

            public Tensor extract_glimpse(Tensor input, Tensor size, Tensor offsets, bool centered = true, bool normalized = true,
                bool uniform_noise = true, string name = null)
                => image_ops_impl.extract_glimpse(input, size, offsets, centered, normalized, uniform_noise, name);

            public (Tensor, Tensor, Tensor, Tensor) combined_non_max_suppression(Tensor boxes, Tensor scores, Tensor max_output_size_per_class,
                Tensor max_total_size, float iou_threshold = 0.5f, float score_threshold = -1f / 0f, bool pad_per_class = false, bool clip_boxes = true,
                string name = null)
                => image_ops_impl.combined_non_max_suppression(boxes, scores, max_output_size_per_class, max_total_size, iou_threshold, score_threshold,
                    pad_per_class, clip_boxes, name);

            public (Tensor, Tensor) non_max_suppression_padded(Tensor boxes, Tensor scores, Tensor max_output_size,
                float iou_threshold = 0.5f,
                float score_threshold = -1f / 0f,
                bool pad_to_max_output_size = false,
                string name = null,
                bool sorted_input = false,
                bool canonicalized_coordinates = false,
                int tile_size = 512)
                => image_ops_impl.non_max_suppression_padded(boxes, scores, max_output_size, iou_threshold, score_threshold, pad_to_max_output_size,
                    name, sorted_input, canonicalized_coordinates, tile_size);

            public Tensor resize(Tensor image, Shape size, string method = ResizeMethod.BILINEAR)
                => image_ops_impl.resize_images_v2(image, size, method: method);

            public Tensor resize(Tensor image, Tensor size, string method = ResizeMethod.BILINEAR)
                => image_ops_impl.resize_images_v2(image, size, method: method);

            public Tensor resize_bilinear(Tensor images, Tensor size, bool align_corners = false, bool half_pixel_centers = false, string name = null)
                => gen_image_ops.resize_bilinear(images, size, align_corners: align_corners, half_pixel_centers: half_pixel_centers, name: name);

            public Tensor resize_images(Tensor images, Tensor size, string method = ResizeMethod.BILINEAR,
                    bool preserve_aspect_ratio = false, string name = null)
                => image_ops_impl.resize_images(images, size, method: method,
                    preserve_aspect_ratio: preserve_aspect_ratio, name: name);

            public Tensor convert_image_dtype(Tensor image, TF_DataType dtype, bool saturate = false, string name = null)
                => gen_image_ops.convert_image_dtype(image, dtype, saturate: saturate, name: name);

            public Tensor decode_image(Tensor contents, int channels = 0, TF_DataType dtype = TF_DataType.TF_UINT8,
                string name = null, bool expand_animations = true)
                => image_ops_impl.decode_image(contents, channels: channels, dtype: dtype,
                    name: name, expand_animations: expand_animations);

            public Tensor encode_png(Tensor contents, string name = null)
                    => image_ops_impl.encode_png(contents, name: name);

            public Tensor encode_jpeg(Tensor contents, string name = null)
                    => image_ops_impl.encode_jpeg(contents, name: name);


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

            public Tensor draw_bounding_boxes(Tensor images, Tensor boxes, Tensor colors = null, string name = null)
                => image_ops_impl.draw_bounding_boxes(images, boxes, colors, name);
        }
    }
}
