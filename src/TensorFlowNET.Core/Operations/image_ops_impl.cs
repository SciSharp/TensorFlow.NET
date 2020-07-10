﻿/*****************************************************************************
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
using System.Linq;
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

        internal static Operation[] _CheckAtLeast3DImage(Tensor image, bool require_static)
        {
            TensorShape image_shape;
            try
            {
                if ( image.shape.NDims == null )
                {
                    image_shape = image.shape.with_rank(3);
                } else {
                    image_shape = image.shape.with_rank_at_least(3);
                }
            }
            catch (ValueError)
            {
                throw new ValueError("'image' must be at least three-dimensional.");
            }
            if ( require_static &! image_shape.is_fully_defined() )
            {
                throw new ValueError("\'image\' must be fully defined.");
            }
            foreach (int x in image_shape[-3..])
            {
                throw new ValueError("inner 3 dims of \'image.shape\' must be > 0: %s" %
                                    image_shape);
            }
            if ( !image_shape[-3..].is_fully_defined() )
            {
                return new Operation[] {
                    check_ops.assert_positive(
                        array_ops.shape(image)[-3..],
                        new {@"inner 3 dims of 'image.shape'
                         must be > 0."}),
                    check_ops.assert_greater_equal(
                        array_ops.rank(image),
                        ops.convert_to_tensor(3),
                        message: "'image' must be at least three-dimensional.")
                };
            } else {
                return new Operation[] {};
            }
        }

        internal static Tensor fix_image_flip_shape(Tensor image, Tensor result)
        {
            TensorShape image_shape = image.shape;
            if (image_shape == tensor_shape.unknown_shape())
            {
                result.set_shape(new { null, null, null });
            } else {
                result.set_shape(image_shape);
            }
            return result;
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

        internal static Tensor _random_flip(Tensor image, int flip_index, int seed,
            string scope_name)
        {
            using ( var scope = ops.name_scope(null, scope_name, new { image }) )
            {
                image = ops.convert_to_tensor(image, name: "image");
                image = _AssertAtLeast3DImage(image);
                Tensor shape = image.shape;
                if ( shape.NDims == 3 || shape.NDims == null )
                {
                    var uniform_random = random_ops.random_uniform(new {}, 0, 1.0, seed: seed);
                    var mirror_cond = math_ops.less(uniform_random, .5);
                    var result = control_flow_ops.cond(
                        pred: mirror_cond,
                        true_fn: array_ops.reverse(image, new { flip_index }),
                        false_fn: image,
                        name: scope
                    );
                    return fix_image_flip_shape(image, result);
                } else if ( shape.NDims == 4 )
                {
                    var batch_size = array_ops.shape(image);
                    var uniform_random = random_ops.random_uniform(batch_size[0],
                                                                    0,
                                                                    1.0 as float,
                                                                    seed: seed);
                    var flips = math_ops.round(
                        array_ops.reshape(uniform_random, shape: new Tensor [batch_size[0], 1, 1, 1]));
                    flips = math_ops.cast(flips, image.dtype);
                    var flipped_input = array_ops.reverse(image, new { flip_index + 1 });
                    return flips * flipped_input + (1 - flips) * image;
                } else
                {
                    throw new ValueError("'\'image\' must have either 3 or 4 dimensions.");
                }
            }
        }

        public static Tensor flip_left_right(Tensor image)
            => _flip(image, 1, "flip_left_right");

        public static Tensor flip_up_down(Tensor image)
            => _flip(image, 1, "flip_up_down");

        internal static Tensor _flip(Tensor image, int flip_index, string scope_name)
        {
            return tf_with(ops.name_scope(null, scope_name, new { image }), delegate
            {
                image = ops.convert_to_tensor(image, name: "image");
                image = _AssertAtLeast3DImage(image);
                Tensor shape = image.shape;
                if ( shape.NDims == 3 || shape.NDims == null )
                {
                    return fix_image_flip_shape(image, array_ops.reverse(image, new { flip_index }));
                } else if ( shape.NDims == 4 )
                {
                    return array_ops.reverse(image, new { flip_index + 1 });
                } else
                {
                    throw new ValueError("\'image\' must have either 3 or 4 dimensions.");
                }
            });
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
                var signature = tf.strings.substr(contents, 0, 2);
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
                substr = tf.strings.substr(contents, 0, 3);
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
                var substr = tf.strings.substr(contents, 0, 3);
                var jpg = Encoding.UTF8.GetString(new byte[] { 0xff, 0xd8, 0xff });
                var jpg_tensor = tf.constant(jpg);
                var result = math_ops.equal(substr, jpg_tensor, name: name);
                return result;
            });
        }

        public static Tensor _is_png(Tensor contents, string name = null)
        {
            return tf_with(ops.name_scope(name, "is_png"), scope =>
            {
                var substr = tf.strings.substr(contents, 0, 3);
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
