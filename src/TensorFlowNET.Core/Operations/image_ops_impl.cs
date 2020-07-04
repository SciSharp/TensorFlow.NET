/*****************************************************************************
   Copyright 2020 Haiping Chen. All Rights Reserved.

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
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class image_ops_impl
    {
        internal static Tensor _AssertAtLeast3DImage(Tensor image)
            => control_flow_ops.with_dependencies(
                _CheckAtLeast3DImage(image, require_static: false), image);

        internal static Array _CheckAtLeast3DImage(Tensor image, bool require_static)
        {
            throw new NotImplementedException("");
        }

        public static Tensor random_flip_up_down(Tensor image, int seed = 0)
            => _random_flip(image: image,
                            flip_index: 0, 
                            seed: seed, 
                            scope_name: "random_flip_up_down");

        public static Tensor random_flip_left_right(Tensor image, int seed = 0)
            => _random_flip(image: image,
                            flip_index: 1,
                            seed: seed,
                            scope_name: "random_flip_left_right");

        internal static Tensor _random_flip(Tensor image, int flipindex, int seed,
            string scope_name)
        {
            using ( scope = ops.name_scope(null, scope_name, image))
            {
                image = ops.convert_to_tensor(image, name: "image");
                image = AssertAtLeast3DImage(image);
                var shape = image.get_shape();
                if ( shape.NDims == 3 || shape.NDims == null )
                {
                    var uniform_random = random_ops.random_uniform(new Tensor [], 0, 1.0, seed: seed);
                    var mirror_cond = math_ops.less(uniform_random, .5);
                    var result = control_flow_ops.cond(
                        pred: mirror_cond,
                        true_fn: array_ops.reverse(image, flipindex as int[]),
                        false_fn: image,
                        name: scope
                    );
                    return fix_image_flip_shape(image, result);
                } else if ( shape.NDims == 4 )
                {
                    var batch_size = array_ops.shape(image)[0];
                    var uniform_random = random_ops.random_uniform(batch_size,
                                                                    0,
                                                                    1.0,
                                                                    seed: seed);
                    var flips = math_ops.round(
                        array_ops.reshape(uniform_random, shape: new Tensor [batch_size, 1, 1, 1]));
                    flips = math_ops.cast(flips, image.dtype);
                    var flipped_input = array_ops.reverse(image, flip_index + 1 as int[]);
                    return flips * flipped_input + (1 - flips) * image;
                } else
                {
                    throw new ValueError("'\'image\' must have either 3 or 4 dimensions.");
                }
            }
        }
       
        public static Tensor decode_image(Tensor contents, int channels = 0, TF_DataType dtype = TF_DataType.TF_UINT8,
            string name = null, bool expand_animations = true)
        {
            Tensor substr = null;

            Func<ITensorOrOperation> _jpeg = () =>
            {
                int jpeg_channels = channels;
                var good_channels = math_ops.not_equal(jpeg_channels, 4, name: "check_jpeg_channels");
                string channels_msg = "Channels must be in (None, 0, 1, 3) when decoding JPEG 'images'";
                var assert_channels = control_flow_ops.Assert(good_channels, new string[] { channels_msg });
                return tf_with(ops.control_dependencies(new[] { assert_channels }), delegate
                {
                    return convert_image_dtype(gen_image_ops.decode_jpeg(contents, channels), dtype);
                });
            };

            Func<ITensorOrOperation> _gif = () =>
            {
                int gif_channels = channels;
                var good_channels = math_ops.logical_and(
                  math_ops.not_equal(gif_channels, 1, name: "check_gif_channels"),
                  math_ops.not_equal(gif_channels, 4, name: "check_gif_channels"));

                string channels_msg = "Channels must be in (None, 0, 3) when decoding GIF images";
                var assert_channels = control_flow_ops.Assert(good_channels, new string[] { channels_msg });
                return tf_with(ops.control_dependencies(new[] { assert_channels }), delegate
                {
                    var result = convert_image_dtype(gen_image_ops.decode_gif(contents), dtype);
                    if (!expand_animations)
                        // result = array_ops.gather(result, 0);
                        throw new NotImplementedException("");
                    return result;
                });
            };

            Func<ITensorOrOperation> _bmp = () =>
            {
                int bmp_channels = channels;
                var signature = string_ops.substr(contents, 0, 2);
                var is_bmp = math_ops.equal(signature, "BM", name: "is_bmp");
                string decode_msg = "Unable to decode bytes as JPEG, PNG, GIF, or BMP";
                var assert_decode = control_flow_ops.Assert(is_bmp, new string[] { decode_msg });
                var good_channels = math_ops.not_equal(bmp_channels, 1, name: "check_channels");
                string channels_msg = "Channels must be in (None, 0, 3) when decoding BMP images";
                var assert_channels = control_flow_ops.Assert(good_channels, new string[] { channels_msg });
                return tf_with(ops.control_dependencies(new[] { assert_decode, assert_channels }), delegate
                {
                    return convert_image_dtype(gen_image_ops.decode_bmp(contents), dtype);
                });
            };

            Func<ITensorOrOperation> _png = () =>
            {
                return convert_image_dtype(gen_image_ops.decode_png(
                      contents,
                      channels,
                      dtype: dtype),
                      dtype);
            };

            Func<ITensorOrOperation> check_gif = () =>
            {
                var is_gif = math_ops.equal(substr, "\x47\x49\x46", name: "is_gif");
                return control_flow_ops.cond(is_gif, _gif, _bmp, name: "cond_gif");
            };

            Func<ITensorOrOperation> check_png = () =>
            {
                return control_flow_ops.cond(_is_png(contents), _png, check_gif, name: "cond_png");
            };

            return tf_with(ops.name_scope(name, "decode_image"), scope =>
            {
                substr = string_ops.substr(contents, 0, 3);
                return control_flow_ops.cond(is_jpeg(contents), _jpeg, check_png, name: "cond_jpeg");
            });
        }

        internal static Tensor resize_images(Tensor images, Tensor size, ResizeMethod method, bool align_corners, bool preserve_aspect_ratio, string name)
        {
            throw new NotImplementedException();
        }

        
        public static Tensor crop_and_resize(Tensor image, Tensor boxes, Tensor box_ind, Tensor crop_size, string method, float extrapolation_value, string name)
        {
            var _op = tf._op_def_lib._apply_op_helper("CropAndResize", name: name, args: new
            {
                image,
                boxes,
                box_ind,
                crop_size,
                method,
                extrapolation_value
            });

            return _op.outputs[0];
        }

        public static Tensor is_jpeg(Tensor contents, string name = null)
        {
            return tf_with(ops.name_scope(name, "is_jpeg"), scope =>
            {
                var substr = string_ops.substr(contents, 0, 3);
                return math_ops.equal(substr, "\xff\xd8\xff", name: name);
            });
        }

        public static Tensor _is_png(Tensor contents, string name = null)
        {
            return tf_with(ops.name_scope(name, "is_png"), scope =>
            {
                var substr = string_ops.substr(contents, 0, 3);
                return math_ops.equal(substr, @"\211PN", name: name);
            });
        }

        public static Tensor convert_image_dtype(Tensor image, TF_DataType dtype, bool saturate = false, 
            string name = null)
        {
            if (dtype == image.dtype)
                return array_ops.identity(image, name: name);

            throw new NotImplementedException("");
        }

        /// <summary>
        /// Resize `images` to `size` using nearest neighbor interpolation.
        /// </summary>
        /// <param name="images"></param>
        /// <param name="size"></param>
        /// <param name="align_corners"></param>
        /// <param name="name"></param>
        /// <param name="half_pixel_centers"></param>
        /// <returns></returns>
        public static Tensor resize_nearest_neighbor<Tsize>(Tensor images, Tsize size, bool align_corners = false, 
            string name = null, bool half_pixel_centers = false)
            => gen_image_ops.resize_nearest_neighbor(images: images,
                  size: size,
                  align_corners: align_corners,
                  half_pixel_centers: half_pixel_centers,
                  name: name);
    }

    public enum ResizeMethod
    {
        BILINEAR = 0,
        NEAREST_NEIGHBOR = 1,
        BICUBIC = 2,
        AREA = 3
    }
}
