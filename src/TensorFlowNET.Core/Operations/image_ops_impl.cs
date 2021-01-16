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
using System.Linq;
using Tensorflow.Framework;
using Tensorflow.Operations;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class image_ops_impl
    {
        internal static Operation _assert(Tensor cond, Type ex_type, string msg)
        {
            if (_is_tensor(cond))
                return control_flow_ops.Assert(cond, new object[] { msg });
            else
                if (cond != null)
            {
                Exception ex_type2 = (Exception)Activator.CreateInstance(ex_type, msg, ex_type);
                throw ex_type2;
            }
            else
            {
                Operation x = null;
                return x;
            }
        }

        internal static bool _is_tensor(object x)
        {
            if (isinstance(x, typeof(Tensor)))
                return true;
            else if (isinstance(x, typeof(IVariableV1)))
                return true;
            else
                return false;
        }

        internal static int[] _ImageDimensions(Tensor image, int rank)
        {
            if (image.TensorShape.is_fully_defined())
                return image.TensorShape.as_list();
            else
            {
                var static_shape = image.TensorShape.with_rank(rank).as_list();
                var dynamic_shape = array_ops.unstack(array_ops.shape(image), rank);

                int[] ss_storage = null;
                int[] ds_storage = null;
                // var sd = static_shape.Zip(dynamic_shape, (first, second) => storage[storage.Length] = first;
                var sd = static_shape.Zip(dynamic_shape, (ss, ds) =>
                {
                    ss_storage[ss_storage.Length] = ss;
                    ds_storage[ds_storage.Length] = (int)ds;
                    return true;
                });

                if (ss_storage != null)
                    return ss_storage;
                else
                    return ds_storage;
            }
        }

        internal static Tensor _AssertAtLeast3DImage(Tensor image)
            => control_flow_ops.with_dependencies(_CheckAtLeast3DImage(image, require_static: false), image);

        internal static Operation[] _CheckAtLeast3DImage(Tensor image, bool require_static)
        {
            TensorShape image_shape;
            try
            {
                if (image.TensorShape.ndim == Unknown)
                {
                    image_shape = image.TensorShape.with_rank(3);
                }
                else
                {
                    image_shape = image.TensorShape.with_rank_at_least(3);
                }
            }
            catch (ValueError)
            {
                throw new ValueError("'image' must be at least three-dimensional.");
            }
            if (require_static & !image_shape.is_fully_defined())
            {
                throw new ValueError("\'image\' must be fully defined.");
            }
            for (int x = 1; x < 4; x++)
            {
                if (image_shape.dims[x] == 0)
                {
                    throw new ValueError(String.Format("inner 3 dims of \'image.shape\' must be > 0: {0}", image_shape));
                }
            }

            var image_shape_last_three_elements = new TensorShape(new int[3] {
                                                image_shape.dims[image_shape.dims.Length - 1],
                                                image_shape.dims[image_shape.dims.Length - 2],
                                                image_shape.dims[image_shape.dims.Length - 3]});
            if (!image_shape_last_three_elements.is_fully_defined())
            {
                Tensor image_shape_ = array_ops.shape(image);
                var image_shape_return = tf.constant(new int[3] {
                    image_shape_.dims[image_shape.dims.Length - 1],
                    image_shape_.dims[image_shape.dims.Length - 2],
                    image_shape_.dims[image_shape.dims.Length - 3]});

                return new Operation[] {
                    check_ops.assert_positive(
                        image_shape_return,
                        new object[] {"inner 3 dims of 'image.shape must be > 0."}
                    ),
                    check_ops.assert_greater_equal(
                        x: array_ops.rank(image),
                        y: tf.constant(3),
                        message: "'image' must be at least three-dimensional."
                    )
                };
            }
            else
            {
                return new Operation[] { };
            }
        }

        internal static Tensor fix_image_flip_shape(Tensor image, Tensor result)
        {
            TensorShape image_shape = image.shape;
            if (image_shape == image_shape.unknown_shape())
            {
                // c# defaults null types to 0 anyhow, so this should be a pretty equivalent port
                result.set_shape(new TensorShape(new int[] { 0, 0, 0 }));
            }
            else
            {
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

        internal static Tensor _random_flip(Tensor image, int flip_index, int seed, string scope_name)
        {
            return tf_with(ops.name_scope(null, scope_name, new[] { image }), scope =>
             {
                 image = ops.convert_to_tensor(image, name: "image");
                 image = _AssertAtLeast3DImage(image);
                 TensorShape shape = image.shape;
                 if (shape.ndim == 3 || shape.ndim == Unknown)
                 {
                     Tensor uniform_random = random_ops.random_uniform(new int[] { }, 0f, 1.0f, seed: seed);
                     var mirror_cond = gen_math_ops.less(uniform_random, .5);

                     var result = control_flow_ops.cond(
                         pred: mirror_cond,
                         true_fn: () => gen_array_ops.reverse(image, new { flip_index }),
                         false_fn: () => image,
                         name: scope
                     );
                     return fix_image_flip_shape(image, result);
                 }
                 else if (shape.ndim == 4)
                 {
                     var batch_size = array_ops.shape(image);
                     var uniform_random = random_ops.random_uniform(batch_size.shape,
                                                                     0f,
                                                                     1.0f,
                                                                     seed: seed);
                     var flips = math_ops.round(
                         array_ops.reshape(uniform_random, shape: array_ops.constant(value: new object[] { batch_size[0], 1, 1, 1 })));
                     flips = math_ops.cast(flips, image.dtype);
                     var flipped_input = gen_array_ops.reverse(image, new int[] { flip_index + 1 });
                     return flips * flipped_input + (1 - flips) * image;
                 }
                 else
                 {
                     throw new ValueError(String.Format("\'image\' {0} must have either 3 or 4 dimensions.", shape));
                 }
             });
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
                  TensorShape shape = image.shape;
                  if (shape.ndim == 3 || shape.ndim == Unknown)
                  {
                      return fix_image_flip_shape(image, gen_array_ops.reverse(image, new { flip_index }));
                  }
                  else if (shape.ndim == 4)
                  {
                      return gen_array_ops.reverse(image, new[] { flip_index + 1 });
                  }
                  else
                  {
                      throw new ValueError("\'image\' must have either 3 or 4 dimensions.");
                  }
              });
        }

        public static Tensor rot90(Tensor image, int k = 1, string name = null)
        {
            return tf_with(ops.name_scope(name, "rot90", new[] { image, tf.constant(k) }), scope =>
             {
                 image = ops.convert_to_tensor(image, name: "image");
                 image = _AssertAtLeast3DImage(image);

                 // can't get k to convert to tensor without throwing error about it being an int---
                 // might rework later. for now, k2 == k as Tensor
                 Tensor k2 = ops.convert_to_tensor(k, dtype: dtypes.int32, name: "k");
                 k2.TensorShape.assert_has_rank(0);
                 k2 = gen_ops.mod(k2, tf.constant(4));

                 TensorShape shape = image.shape;
                 if (shape.ndim == 3 || shape.ndim == Unknown)
                 {
                     return _rot90_3D(image, k, scope);
                 }
                 else if (shape.ndim == 4)
                 {
                     return _rot90_3D(image, k, scope);
                 }
                 else
                 {
                     throw new ValueError(String.Format("\'image\' {0} must have either 3 or 4 dimensions.", shape));
                 }
             });
        }

        internal static Tensor _rot90_3D(Tensor image, int k, string name_scope)
        {
            Tensor _rot90()
            {
                return array_ops.transpose(gen_array_ops.reverse(image, new[] { 1, 0, 2 }), new int[] { 1 });
            };
            Tensor _rot180()
            {
                return gen_array_ops.reverse(image, new[] { 0, 1 });
            };
            Tensor _rot270()
            {
                return gen_array_ops.reverse(array_ops.transpose(image, new[] { 1, 0, 2 }), new[] { 1 });
            };

            var cases = new[] {math_ops.equal(k, 1), _rot90(),
                                math_ops.equal(k, 2), _rot180(),
                                math_ops.equal(k, 3), _rot270()};

            var result = control_flow_ops.case_v2(cases, callable_default: () => new Tensor[] { image }, exclusive: true, name: name_scope);
            result.set_shape(new[] { -1, -1, image.TensorShape.dims[2] });
            return result;
        }

        public static Tensor transpose(Tensor image, string name = null)
        {
            using (ops.name_scope(name, "transpose", new[] { image }))
                return tf_with(ops.name_scope(name, "transpose", new[] { image }), delegate
                 {
                     image = ops.convert_to_tensor(image, name: "image");
                     image = _AssertAtLeast3DImage(image);
                     TensorShape shape = image.shape;
                     if (shape.ndim == 3 || shape.ndim == Unknown)
                     {
                         return array_ops.transpose(image, new[] { 1, 0, 2 }, name: name);
                     }
                     else if (shape.ndim == 4)
                     {
                         return array_ops.transpose(image, new[] { 0, 2, 1, 3 }, name: name);
                     }
                     else
                     {
                         throw new ValueError(String.Format("\'image\' {0} must have either 3 or 4 dimensions."));
                     }
                 });
        }

        public static Tensor central_crop(Tensor image, float central_fraction)
        {
            using (ops.name_scope(null, "central_crop", new[] { image }))
            {
                image = ops.convert_to_tensor(image, name: "image");
                if (central_fraction <= 0.0 || central_fraction > 1.0)
                    throw new ValueError("central_fraction must be within (0, 1]");
                if (central_fraction == 1.0)
                    return image;

                _AssertAtLeast3DImage(image);
                var rank = image.TensorShape.ndim;
                if (rank != 3 && rank != 4)
                    throw new ValueError(String.Format(@"`image` should either be a Tensor with rank = 3
or rank = 4. Had rank = {0}", rank));

                object[] _get_dim(Tensor tensor, int idx)
                {
                    var static_shape = tensor.TensorShape.dims[idx];
                    if (static_shape != (int)None)
                        return new object[2] { static_shape, false };
                    return new object[2] { array_ops.shape(tensor)[idx], true };
                };

                object[] h, w;
                int d, bs = 0;
                if (rank == 3)
                {
                    h = _get_dim(image, 0); // img_h == h[0], dynamic_h == h[1]
                    w = _get_dim(image, 1);
                    d = image.shape[3];
                }
                else
                {
                    bs = image.shape[0];
                    h = _get_dim(image, 1);
                    w = _get_dim(image, 2);
                    d = image.shape[3];
                }

                object hd, bbox_h_start;
                if ((bool)h[1])
                {
                    hd = math_ops.cast((IVariableV1)h[0], dtypes.float64);
                    bbox_h_start = math_ops.cast(((int)hd - (int)hd * central_fraction) / 2, dtypes.int32);
                }
                else
                {
                    hd = (float)w[0];
                    bbox_h_start = (int)(((int)hd - (int)hd * central_fraction) / 2);
                }

                object wd, bbox_w_start;
                if ((bool)w[1])
                {
                    wd = math_ops.cast((IVariableV1)w[0], dtypes.float64);
                    bbox_w_start = math_ops.cast(((int)wd - (int)wd * central_fraction) / 2, dtypes.int32);
                }
                else
                {
                    wd = (float)w[0];
                    bbox_w_start = (int)(((int)wd - (int)wd * central_fraction) / 2);
                }

                var bbox_h_size = (int)h[0] - (int)bbox_h_start * 2;
                var bbox_w_size = (int)w[0] - (int)bbox_w_start * 2;

                Tensor bbox_begin, bbox_size;
                if (rank == 3)
                {
                    bbox_begin = array_ops.stack(ops.convert_to_tensor(new[] { bbox_h_start, bbox_w_start, 0 }));
                    bbox_size = array_ops.stack(ops.convert_to_tensor(new[] { bbox_h_size, bbox_w_size, -1 }));
                }
                else
                {
                    bbox_begin = array_ops.stack(ops.convert_to_tensor(new[] { 0, bbox_h_start, bbox_w_start, 0 }));
                    bbox_size = array_ops.stack(ops.convert_to_tensor(new[] { -1, bbox_h_size, bbox_w_size, -1 }));
                }

                image = array_ops.slice(image, bbox_begin, bbox_size);

                int arg1()
                {
                    if ((bool)h[1])
                    {
                        // 0 == null for nullable ints anyways
                        return 0;
                    }
                    else
                    {
                        return bbox_h_size;
                    }
                };
                int arg2()
                {
                    if ((bool)w[1])
                    {
                        return 0;
                    }
                    else
                    {
                        return bbox_w_size;
                    }
                };
                if (rank == 3)
                {
                    var _arg1 = arg1();
                    var _arg2 = arg2();

                    image.set_shape(ops.convert_to_tensor(new object[
                        _arg1, _arg2, d
                    ]));
                }
                else
                {
                    var _arg1 = arg1();
                    var _arg2 = arg2();
                    image.set_shape(ops.convert_to_tensor(new object[] {
                        bs, _arg1, _arg2, d
                    }));
                }
            }

            return image;
        }

        public static Tensor pad_to_bounding_box(Tensor image, int offset_height, int offset_width,
            int target_height, int target_width)
        {
            return tf_with(ops.name_scope(null, "pad_to_bounding_box", new[] { image }), delegate
             {
                 image = ops.convert_to_tensor(image, name: "image");

                 bool is_batch = true;
                 TensorShape image_shape = image.shape;
                 if (image_shape.ndim == 3)
                 {
                     is_batch = false;
                     image = array_ops.expand_dims(image, 0);
                 }
                 else if (image_shape.ndim == Unknown)
                 {
                     is_batch = false;
                     image = array_ops.expand_dims(image, 0);
                     image.set_shape(new TensorShape(0, 0, 0, 0));
                 }
                 else if (image_shape.ndim != 4)
                 {
                     throw new ValueError(String.Format("\'image\' {0} must have either 3 or 4 dimensions.",
                         image_shape));
                 }

                 var assert_ops = _CheckAtLeast3DImage(image, require_static: false);

                 // batch: [0], height: [1], width: [2], depth: [3]
                 int[] bhwd = _ImageDimensions(image, rank: 4);

                 var after_padding_width = target_width - offset_width - bhwd[2];

                 var after_padding_height = target_height - offset_height - bhwd[1];

                 assert_ops[assert_ops.Length] = _assert(check_ops.assert_greater_equal(tf.constant(offset_height),
                                                         tf.constant(0)), typeof(ValueError),
                                                         "offset_height must be >= 0");
                 assert_ops[assert_ops.Length] = _assert(check_ops.assert_greater_equal(tf.constant(offset_width),
                                                         tf.constant(0)), typeof(ValueError),
                                                         "offset_width must be >= 0");
                 assert_ops[assert_ops.Length] = _assert(check_ops.assert_greater_equal(tf.constant(after_padding_width),
                                                         tf.constant(0)), typeof(ValueError),
                                                         "width must be <= target - offset");
                 assert_ops[assert_ops.Length] = _assert(check_ops.assert_greater_equal(tf.constant(after_padding_height),
                                                         tf.constant(0)), typeof(ValueError),
                                                         "height must be <= target - offset");
                 image = control_flow_ops.with_dependencies(assert_ops, image);

                 var paddings = array_ops.reshape(
                     array_ops.stack(new[] {
                        0, 0, offset_height, after_padding_height, offset_width,
                        after_padding_width, 0, 0
                     }), new[] { 4, 2 }
                 );
                 var padded = array_ops.pad(image, paddings);

                 TensorShape padded_shape_result()
                 {
                     int[] i_remnants = { };
                     foreach (var i in new[] { bhwd[0], target_height, target_width, bhwd[3] })
                         if (_is_tensor(i))
                             return null;
                         else
                             i_remnants[i_remnants.Length] = i;
                     return new TensorShape(i_remnants);
                 };
                 TensorShape padded_shape = padded_shape_result();
                 padded.set_shape(padded_shape);

                 if (!is_batch)
                 {
                     padded = array_ops.squeeze(padded, axis: new int[] { 0 });
                 }

                 return padded;
             });
        }

        public static Tensor crop_to_bounding_box(Tensor image, int offset_height, int offset_width,
            int target_height, int target_width)
        {
            return tf_with(ops.name_scope(null, "crop_to_bounding_box", new[] { image }), delegate
             {
                 image = ops.convert_to_tensor(image, name: "image");

                 bool is_batch = true;
                 TensorShape image_shape = image.shape;
                 if (image_shape.ndim == 3)
                 {
                     is_batch = false;
                     image = array_ops.expand_dims(image, 0);
                 }
                 else if (image_shape.ndim == Unknown)
                 {
                     is_batch = false;
                     image = array_ops.expand_dims(image, 0);
                     image.set_shape(new TensorShape(new int[] { 0, 0, 0, 0 }));
                 }
                 else if (image_shape.ndim != 4)
                 {
                     throw new ValueError(String.Format("\'image\' {0} must have either 3 or 4 dimensions.",
                         image_shape));
                 }

                 var assert_ops = _CheckAtLeast3DImage(image, require_static: false);

                 // batch: [0], height: [1], width: [2], depth: [3]
                 int[] bhwd = _ImageDimensions(image, rank: 4);

                 assert_ops[assert_ops.Length] = _assert(check_ops.assert_greater_equal(tf.constant(offset_height),
                                                         tf.constant(0)), typeof(ValueError),
                                                         "offset_height must be >= 0.");
                 assert_ops[assert_ops.Length] = _assert(check_ops.assert_greater_equal(tf.constant(offset_width),
                                                         tf.constant(0)), typeof(ValueError),
                                                         "offset_width must be >= 0.");
                 assert_ops[assert_ops.Length] = _assert(check_ops.assert_less(tf.constant(0),
                                                         tf.constant(target_width)), typeof(ValueError),
                                                         "target_width must be > 0.");
                 assert_ops[assert_ops.Length] = _assert(check_ops.assert_less(tf.constant(0),
                                                         tf.constant(target_height)), typeof(ValueError),
                                                         "target_height must be > 0.");
                 assert_ops[assert_ops.Length] = _assert(check_ops.assert_greater_equal(tf.constant(bhwd[2]),
                                                         tf.constant(target_width + offset_width)),
                                                         typeof(ValueError),
                                                         "width must be >= target + offset.");
                 assert_ops[assert_ops.Length] = _assert(check_ops.assert_greater_equal(tf.constant(bhwd[1]),
                                                         tf.constant(target_height + offset_height)),
                                                         typeof(ValueError),
                                                         "height must be >= target + offset.");
                 image = control_flow_ops.with_dependencies(assert_ops, image);

                 var cropped = array_ops.slice(
                     image, array_ops.stack(new[] { 0, offset_height, offset_width, 0 }),
                     array_ops.stack(new[] { -1, target_height, target_width, -1 }));

                 TensorShape cropped_shape_result()
                 {
                     int[] i_remnants = { };
                     foreach (var i in new[] { bhwd[0], target_height, target_width, bhwd[3] })
                         if (_is_tensor(i))
                             return null;
                         else
                             i_remnants[i_remnants.Length] = i;
                     return new TensorShape(i_remnants);
                 };
                 var cropped_shape = cropped_shape_result();
                 cropped.set_shape(cropped_shape);

                 if (!is_batch)
                 {
                     cropped = array_ops.squeeze(cropped, axis: new int[] { 0 });
                 }

                 return cropped;
             });
        }

        public static Tensor resize_image_with_crop_or_pad(Tensor image, object target_height, object target_width)
        {
            using (ops.name_scope(null, "resize_image_with_crop_or_pad", new[] { image }))
                return tf_with(ops.name_scope(null, "resize_image_with_crop_or_pad", new[] { image }), delegate
                 {
                     image = ops.convert_to_tensor(image, name: "image");
                     TensorShape image_shape = image.shape;
                     bool is_batch = true;
                     if (image_shape.ndim == 3)
                     {
                         is_batch = false;
                         image = array_ops.expand_dims(image, 0);
                     }
                     else if (image_shape.ndim == Unknown)
                     {
                         is_batch = false;
                         image = array_ops.expand_dims(image, 0);
                         image.set_shape(new TensorShape(new int[] { 0, 0, 0, 0 }));
                     }
                     else if (image_shape.ndim != 4)
                     {
                         throw new ValueError(String.Format("\'image\' {0} must have either 3 or 4 dimensions.",
                             image_shape));
                     }

                     var assert_ops = _CheckAtLeast3DImage(image, require_static: false);
                     assert_ops[assert_ops.Length] = _assert(check_ops.assert_less(tf.constant(0),
                                                             tf.constant(target_width)),
                                                             typeof(ValueError),
                                                             "target_width must be > 0.");
                     assert_ops[assert_ops.Length] = _assert(check_ops.assert_less(tf.constant(0),
                                                             tf.constant(target_height)),
                                                             typeof(ValueError),
                                                             "target_height must be > 0.");

                     image = control_flow_ops.with_dependencies(assert_ops, image);

                     if (_is_tensor(target_height))
                     {
                         target_height = control_flow_ops.with_dependencies(
                             assert_ops, tf.constant(target_height));
                     }
                     if (_is_tensor(target_width))
                     {
                         target_width = control_flow_ops.with_dependencies(
                             assert_ops, tf.constant(target_width));
                     }


                     object max_(object x, object y)
                     {
                         if (_is_tensor(x) || _is_tensor(y))
                             return math_ops.maximum(x, y);
                         else
                             return Math.Max((int)x, (int)y);
                     }

                     object min_(object x, object y)
                     {
                         if (_is_tensor(x) || _is_tensor(y))
                             return math_ops.minimum(x, y);
                         else
                             return Math.Min((int)x, (int)y);
                     }

                     object equal_(object x, object y)
                     {
                         if (_is_tensor(x) || _is_tensor(y))
                             return math_ops.equal(x, y);
                         else
                             return x == y;
                     }

                     int[] _hw_ = _ImageDimensions(image, rank: 4);
                     int width_diff = (int)target_width - _hw_[2];
                     int offset_crop_width = (int)max_(Math.Floor(Math.Abs((decimal)width_diff) / 2), 0);
                     int offset_pad_width = (int)max_(Math.Floor((decimal)width_diff / 2), 0);

                     int height_diff = (int)target_height - _hw_[1];
                     int offset_crop_height = (int)max_(Math.Floor(Math.Abs((decimal)height_diff) / 2), 0);
                     int offset_pad_height = (int)max_(Math.Floor((decimal)height_diff / 2), 0);

                     Tensor cropped = crop_to_bounding_box(image, offset_crop_height, offset_crop_width,
                                                         (int)min_(target_height, _hw_[1]),
                                                         (int)min_(target_width, _hw_[2]));

                     Tensor resized = pad_to_bounding_box(cropped, offset_pad_height, offset_pad_width,
                                                     (int)target_height, (int)target_width);

                     if (resized.TensorShape.ndim == Unknown)
                         throw new ValueError("resized contains no shape.");

                     int[] _rhrw_ = _ImageDimensions(resized, rank: 4);

                     assert_ops = new Operation[2];
                     assert_ops[0] = _assert(
                         (Tensor)equal_(_rhrw_[1], target_height), typeof(ValueError),
                         "resized height is not correct.");
                     assert_ops[1] = _assert(
                         (Tensor)equal_(_rhrw_[2], target_width), typeof(ValueError),
                         "resized width is not correct.");

                     resized = control_flow_ops.with_dependencies(assert_ops, resized);

                     if (!is_batch)
                     {
                         resized = array_ops.squeeze(resized, axis: new int[] { 0 });
                     }

                     return resized;
                 });
        }

        internal static Tensor _resize_images_common(Tensor images, Func<Tensor, Tensor, Tensor> resizer_fn,
            Tensor size, bool preserve_aspect_ratio, string name, bool skip_resize_if_same)
        {
            return tf_with(ops.name_scope(name, "resize", new[] { images, size }), delegate
              {
                  if (images.TensorShape.ndim == Unknown)
                      throw new ValueError("\'images\' contains no shape.");
                  bool is_batch = true;
                  if (images.TensorShape.ndim == 3)
                  {
                      is_batch = false;
                      images = array_ops.expand_dims(images, 0);
                  }
                  else if (images.TensorShape.ndim != 4)
                      throw new ValueError("\'images\' must have either 3 or 4 dimensions.");

                  var (height, width) = (images.dims[1], images.dims[2]);

                  if (!size.TensorShape.is_compatible_with(new[] { 2 }))
                      throw new ValueError(@"\'size\' must be a 1-D Tensor of 2 elements:
new_height, new_width");

                  if (preserve_aspect_ratio)
                  {
                      var _chcw_ = _ImageDimensions(images, rank: 4);

                      var scale_factor_height = (
                          math_ops.cast(size[0], dtypes.float32) /
                          math_ops.cast(_chcw_[1], dtypes.float32));
                      var scale_factor_width = (
                          math_ops.cast(size[1], dtypes.float32) /
                          math_ops.cast(_chcw_[2], dtypes.float32));
                      var scale_factor = math_ops.minimum(scale_factor_height, scale_factor_width);
                      var scaled_height_const = math_ops.cast(
                          math_ops.round(scale_factor *
                                      math_ops.cast(_chcw_[1], dtypes.float32)),
                          dtypes.int32);
                      var scaled_width_const = math_ops.cast(
                          math_ops.round(scale_factor *
                                      math_ops.cast(_chcw_[2], dtypes.float32)),
                          dtypes.int32);

                      size = ops.convert_to_tensor(new[] { scaled_height_const, scaled_width_const },
                                                  dtypes.int32,
                                                  name: "size");
                  }

                  var size_const_as_shape = tensor_util.constant_value_as_shape(size);
                  var new_height_const = tensor_shape.dimension_at_index(size_const_as_shape,
                                                                      0).value;
                  var new_width_const = tensor_shape.dimension_at_index(size_const_as_shape,
                                                                      1).value;

                  bool x_null = true;
                  if (skip_resize_if_same)
                  {
                      foreach (int x in new[] { new_width_const, width, new_height_const, height })
                      {
                          if (width != new_width_const && height == new_height_const)
                          {
                              break;
                          }
                          if (x != 0)
                          {
                              x_null = false;
                          }
                      }
                      if (!x_null)
                          images = array_ops.squeeze(images, axis: new int[] { 0 });
                      return images;
                  }

                  images = resizer_fn(images, size);

                  images.set_shape(new TensorShape(new int[] { Unknown, new_height_const, new_width_const, Unknown }));

                  if (!is_batch)
                      images = array_ops.squeeze(images, axis: new int[] { 0 });
                  return images;
              });
        }

        public static Tensor resize_images(Tensor images, Tensor size, string method = ResizeMethod.BILINEAR,
            bool preserve_aspect_ratio = false, bool antialias = false, string name = null)
        {
            Tensor resize_fn(Tensor images_t, Tensor new_size)
            {
                var scale_and_translate_methods = new string[] {
                    ResizeMethod.LANCZOS3, ResizeMethod.LANCZOS5, ResizeMethod.GAUSSIAN,
                    ResizeMethod.MITCHELLCUBIC
                };

                Tensor resize_with_scale_and_translate(string method)
                {
                    var scale = new Tensor[] {
                        math_ops.cast(new_size, dtype: dtypes.float32),
                        // does this need to be reworked into only elements 1-3 being
                        // passed like it is in the tensorflow code? or does it matter?
                        math_ops.cast(array_ops.shape(images_t), dtype: dtypes.float32)
                    };
                    return gen_ops.scale_and_translate(
                        images_t,
                        new_size,
                        scale,
                        array_ops.zeros(new[] { 2 }),
                        kernel_type: method,
                        antialias: antialias
                    );
                }

                if (method == ResizeMethod.BILINEAR)
                    if (antialias)
                        return resize_with_scale_and_translate("triangle");
                    else
                        return gen_image_ops.resize_bilinear(images_t,
                            new_size,
                            half_pixel_centers: true);
                else if (method == ResizeMethod.NEAREST_NEIGHBOR)
                    return gen_image_ops.resize_nearest_neighbor(images_t,
                        new_size,
                        half_pixel_centers: true);
                else if (method == ResizeMethod.BICUBIC)
                    if (antialias)
                        return resize_with_scale_and_translate("keyscubic");
                    else
                        return gen_image_ops.resize_bicubic(images_t,
                            new_size,
                            half_pixel_centers: true);
                else if (method == ResizeMethod.AREA)
                    return gen_ops.resize_area(images_t, new_size);
                else if (Array.Exists(scale_and_translate_methods, method => method == method))
                    return resize_with_scale_and_translate(method);
                else
                    throw new ValueError(String.Format("Resize method is not implemented: {0}",
                        method));
            }

            return _resize_images_common(
                images,
                resize_fn,
                size,
                preserve_aspect_ratio: preserve_aspect_ratio,
                name: name,
                skip_resize_if_same: false
            );
        }

        internal static Tensor _resize_image_with_pad_common(Tensor image, int target_height, int target_width,
            Func<Tensor, Tensor, Tensor> resize_fn)
        {
            using (ops.name_scope(null, "resize_image_with_pad", new[] { image }))
                return tf_with(ops.name_scope(null, "resize_image_with_pad", new[] { image }), delegate
                 {
                     image = ops.convert_to_tensor(image, name: "tensor");
                     var image_shape = image.TensorShape;
                     bool is_batch = true;
                     if (image_shape.ndim == 3)
                     {
                         is_batch = false;
                         image = array_ops.expand_dims(image, 0);
                     }
                     else if (image_shape.ndim == Unknown)
                     {
                         is_batch = false;
                         image = array_ops.expand_dims(image, 0);
                         image.set_shape(new TensorShape(new[] { Unknown, Unknown, Unknown, Unknown }));
                     }
                     else if (image_shape.ndim != 4)
                     {
                         throw new ValueError(String.Format("\'image\' {0} must have either 3 or 4 dimensions.",
                                                             image_shape));
                     }

                     var assert_ops = _CheckAtLeast3DImage(image, require_static: false);
                     assert_ops[assert_ops.Length] = _assert(check_ops.assert_less(tf.constant(0),
                                                             tf.constant(target_width)),
                                                             typeof(ValueError),
                                                             "target_width must be > 0.");
                     assert_ops[assert_ops.Length] = _assert(check_ops.assert_less(tf.constant(0),
                                                             tf.constant(target_height)),
                                                             typeof(ValueError),
                                                             "target_height must be > 0.");

                     image = control_flow_ops.with_dependencies(assert_ops, image);

                     object max_(object x, object y)
                     {
                         if (_is_tensor(x) || _is_tensor(y))
                             return math_ops.maximum(x, y);
                         else
                             return Math.Max((int)x, (int)y);
                     }

                     var _hw_ = _ImageDimensions(image, rank: 4);

                     var f_height = math_ops.cast(_hw_[1], dtype: dtypes.float32);
                     var f_width = math_ops.cast(_hw_[2], dtype: dtypes.float32);
                     var f_target_height = math_ops.cast(target_height, dtype: dtypes.float32);
                     var f_target_width = math_ops.cast(target_width, dtype: dtypes.float32);

                     var ratio = (Tensor)max_(f_width / f_target_width, f_height / f_target_height);
                     var resized_height_float = f_height / ratio;
                     var resized_width_float = f_width / ratio;
                     var resized_height = math_ops.cast(
                         gen_math_ops.floor(resized_height_float), dtype: dtypes.int32);
                     var resized_width = math_ops.cast(
                         gen_math_ops.floor(resized_width_float), dtype: dtypes.int32);

                     var padding_height = (f_target_height - resized_height_float) / 2;
                     var padding_width = (f_target_width - resized_width_float) / 2;
                     var f_padding_height = gen_math_ops.floor(padding_height);
                     var f_padding_width = gen_math_ops.floor(padding_width);
                     int p_height = (int)max_(0, math_ops.cast(f_padding_height, dtype: dtypes.int32));
                     int p_width = (int)max_(0, math_ops.cast(f_padding_width, dtype: dtypes.int32));

                     var resized = resize_fn(image, new Tensor(new[] { resized_height, resized_width }));

                     var padded = pad_to_bounding_box(resized, p_height, p_width, target_height,
                                                     target_width);

                     if (padded.TensorShape.ndim == Unknown)
                         throw new ValueError("padded contains no shape.");

                     _ImageDimensions(padded, rank: 4);

                     if (!is_batch)
                     {
                         padded = array_ops.squeeze(padded, axis: new int[] { 0 });
                     }

                     return padded;
                 });
        }

        public static Tensor resize_images_with_pad(Tensor image, int target_height, int target_width,
            string method, bool antialias)
        {
            Tensor _resize_fn(Tensor im, Tensor new_size)
            {
                return resize_images(im, new_size, method, antialias: antialias);
            }

            return _resize_image_with_pad_common(image, target_height, target_width,
                                                _resize_fn);
        }

        public static Tensor per_image_standardization(Tensor image)
        {
            return tf_with(ops.name_scope(null, "per_image_standardization", new[] { image }), scope =>
             {
                 image = ops.convert_to_tensor(image, name: "image");
                 image = _AssertAtLeast3DImage(image);

                 var orig_dtype = image.dtype;
                 if (Array.Exists(new[] { dtypes.float16, dtypes.float32 }, orig_dtype => orig_dtype == orig_dtype))
                     image = convert_image_dtype(image, dtypes.float32);

                 var num_pixels_ = array_ops.shape(image).dims;
                 num_pixels_ = num_pixels_.Skip(num_pixels_.Length - 3).Take(num_pixels_.Length - (num_pixels_.Length - 3)).ToArray();
                 Tensor num_pixels = math_ops.reduce_prod(new Tensor(num_pixels_));
                 Tensor image_mean = math_ops.reduce_mean(image, axis: new int[] { -1, -2, -3 }, keepdims: true);

                 var stddev = math_ops.reduce_std(image, axis: new int[] { -1, -2, -3 }, keepdims: true);
                 var min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, image.dtype));
                 var adjusted_stddev = math_ops.maximum(stddev, min_stddev);

                 image = image - image_mean;
                 image = tf.div(image, adjusted_stddev, name: scope); // name: scope in python version
                 return convert_image_dtype(image, orig_dtype, saturate: true);
             });
        }

        public static Tensor random_brightness(Tensor image, float max_delta, int seed = 0)
        {
            if (max_delta < 0)
                throw new ValueError("max_delta must be non-negative.");

            var delta = random_ops.random_uniform(new int[] { }, max_delta * -1, max_delta, seed: seed);
            return adjust_brightness(image, delta);
        }

        public static Tensor random_contrast(Tensor image, float lower, float upper, int seed = 0)
        {
            if (upper <= lower)
                throw new ValueError("upper must be > lower.");

            if (lower < 0)
                throw new ValueError("lower must be non-negative.");

            var contrast_factor = random_ops.random_uniform(new int[] { }, lower, upper, seed: seed);
            return adjust_contrast(image, contrast_factor);
        }

        public static Tensor adjust_brightness(Tensor image, Tensor delta)
        {
            return tf_with(ops.name_scope(null, "adjust_brightness", new[] { image, delta }), name =>
             {
                 image = ops.convert_to_tensor(image, name: "image");
                 var orig_dtype = image.dtype;

                 Tensor flt_image;
                 if (Array.Exists(new[] { dtypes.float16, dtypes.float32 }, orig_dtype => orig_dtype == orig_dtype))
                 {
                     flt_image = image;
                 }
                 else
                 {
                     flt_image = convert_image_dtype(image, dtypes.float32);
                 }

                 var adjusted = math_ops.add(
                     flt_image, math_ops.cast(delta, flt_image.dtype), name: name);

                 return convert_image_dtype(adjusted, orig_dtype, saturate: true);
             });
        }

        public static Tensor adjust_contrast(Tensor images, Tensor contrast_factor)
        {
            return tf_with(ops.name_scope(null, "adjust_brightness", new[] { images, contrast_factor }), name =>
             {
                 images = ops.convert_to_tensor(images, name: "images");
                 var orig_dtype = images.dtype;

                 Tensor flt_images;
                 if (Array.Exists(new[] { dtypes.float16, dtypes.float32 }, orig_dtype => orig_dtype == orig_dtype))
                 {
                     flt_images = images;
                 }
                 else
                 {
                     flt_images = convert_image_dtype(images, dtypes.float32);
                 }

                 var adjusted = gen_ops.adjust_contrastv2(
                     flt_images, contrast_factor: contrast_factor, name: name);

                 return convert_image_dtype(adjusted, orig_dtype, saturate: true);
             });
        }

        public static Tensor adjust_gamma(Tensor image, int gamma = 1, int gain = 1)
        {
            return tf_with(ops.name_scope(null, "adjust_gamma", new[] {image,
                                        tf.constant(gamma), tf.constant(gain)}), name =>
            {
                image = ops.convert_to_tensor(image, name: "image");
                var orig_dtype = image.dtype;

                Tensor flt_image;
                if (Array.Exists(new[] { dtypes.float16, dtypes.float32 }, orig_dtype => orig_dtype == orig_dtype))
                {
                    flt_image = image;
                }
                else
                {
                    flt_image = convert_image_dtype(image, dtypes.float32);
                }

                var assert_op = _assert(ops.convert_to_tensor(gamma >= 0), typeof(ValueError),
                                        "Gamma should be a non-negative real number.");

                // python code has this if as:
                //  `if (assert_op)`
                //
                // given that assert_op is an Operation, that comparison can't be done here,
                // so this just checks to see if it's empty, as that's what _assert returns
                // if it fails to continue down the line of the assert
                Tensor gamma_as_tensor;
                if (assert_op != null)
                    gamma_as_tensor = control_flow_ops.with_dependencies(new[] { assert_op }, tf.constant(gamma));
                else
                    gamma_as_tensor = tf.constant(gamma);

                var adjusted_img = gain * math_ops.pow(flt_image, gamma_as_tensor);

                return convert_image_dtype(adjusted_img, orig_dtype, saturate: true);
            });
        }

        public static Tensor rgb_to_grayscale(Tensor images, string name = null)
        {
            return tf_with(ops.name_scope(name, "rgb_to_grayscale", new[] { images }), name =>
             {
                 images = ops.convert_to_tensor(images, name: "images");
                 var orig_dtype = images.dtype;
                 var flt_image = convert_image_dtype(images, dtypes.float32);

                 var rgb_weights = new Tensor(new double[] { 0.2989, 0.5870, 0.1140 });
                 var gray_float = math_ops.tensordot(flt_image, rgb_weights, new[] { -1, -1 });
                 gray_float = array_ops.expand_dims(gray_float, -1);
                 return convert_image_dtype(gray_float, orig_dtype, name: name);
             });
        }

        public static Tensor grayscale_to_rgb(Tensor images, string name = null)
        {
            return tf_with(ops.name_scope(name, "grayscale_to_rgb", new[] { images }), name =>
             {
                 images = _AssertAtLeast3DImage(images);

                 images = ops.convert_to_tensor(images, name: "images");
                 var rank_1 = array_ops.expand_dims(array_ops.rank(images) - 1, 0);
                 var shape_list = (array_ops.ones(rank_1, dtype: dtypes.int32) +
                                 array_ops.expand_dims(tf.constant(3), 0));
                 var multiples = array_ops.concat(new Tensor[] { shape_list }, 0);
                 var rgb = array_ops.tile(images, multiples, name: name);
                 int[] rgb_temp = images.shape.Take(images.shape.Length - 1).ToArray();
                 rgb.set_shape(array_ops.concat(new Tensor[] { ops.convert_to_tensor(rgb_temp) }, 3));
                 return rgb;
             });
        }

        public static Tensor random_hue(Tensor image, float max_delta, int seed = 0)
        {
            if (max_delta > 0.5)
                throw new ValueError("max_delta must be <= 0.5.");

            if (max_delta < 0)
                throw new ValueError("max_delta must be non-negative.");

            var delta = random_ops.random_uniform(new int[] { }, max_delta * -1, max_delta, seed: seed);
            return adjust_hue(image, delta);
        }

        public static Tensor adjust_hue(Tensor image, Tensor delta, string name = null)
        {
            return tf_with(ops.name_scope(name, "adjust_hue", new[] { image }), name =>
             {
                 image = ops.convert_to_tensor(image, name: "image");
                 var orig_dtype = image.dtype;

                 Tensor flt_image;
                 if (Array.Exists(new[] { dtypes.float16, dtypes.float32 }, orig_dtype => orig_dtype == orig_dtype))
                     flt_image = image;
                 else
                     flt_image = convert_image_dtype(image, dtypes.float32);

                 var rgb_altered = gen_ops.adjust_hue(flt_image, delta);

                 return convert_image_dtype(rgb_altered, orig_dtype);
             });
        }

        public static Tensor random_jpeg_quality(Tensor image, float min_jpeg_quality, float max_jpeg_quality,
            int seed = 0)
        {
            if (min_jpeg_quality < 0 || max_jpeg_quality < 0 || min_jpeg_quality > 100 ||
                max_jpeg_quality > 100)
                throw new ValueError("jpeg encoding range must be between 0 and 100.");

            if (min_jpeg_quality >= max_jpeg_quality)
                throw new ValueError("`min_jpeg_quality` must be less than `max_jpeg_quality`.");

            var jpeg_quality = random_ops.random_uniform(new int[] { },
                                                        min_jpeg_quality,
                                                        max_jpeg_quality,
                                                        seed: seed,
                                                        dtype: dtypes.int32);
            return adjust_jpeg_quality(image, jpeg_quality);
        }

        public static Tensor adjust_jpeg_quality(Tensor image, Tensor jpeg_quality, string name = null)
        {
            return tf_with(ops.name_scope(name, "adjust_jpeg_quality", new[] { image }), delegate
             {
                 image = ops.convert_to_tensor(image, name: "image");
                 var channels = image.TensorShape.as_list()[image.TensorShape.dims.Length - 1];
                 var orig_dtype = image.dtype;
                 // python code checks to ensure jpeq_quality is a tensor; unnecessary here since
                 // it is passed as a tensor
                 image = gen_ops.encode_jpeg_variable_quality(image, quality: jpeg_quality);

                 image = gen_ops.decode_jpeg(image, channels: channels);
                 return convert_image_dtype(image, orig_dtype, saturate: true);
             });
        }

        public static Tensor random_saturation(Tensor image, float lower, float upper, int seed = 0)
        {
            if (upper <= lower)
                throw new ValueError("upper must be > lower.");

            if (lower < 0)
                throw new ValueError("lower must be non-negative");

            var saturation_factor = random_ops.random_uniform(new int[] { }, lower, upper, seed: seed);
            return adjust_saturation(image, saturation_factor);
        }

        public static Tensor adjust_saturation(Tensor image, Tensor saturation_factor, string name = null)
        {
            return tf_with(ops.name_scope(name, "adjust_saturation", new[] { image }), name =>
             {
                 image = ops.convert_to_tensor(image, name: "image");
                 var orig_dtype = image.dtype;

                 Tensor flt_image;
                 if (Array.Exists(new[] { dtypes.float16, dtypes.float32 }, orig_dtype => orig_dtype == orig_dtype))
                     flt_image = image;
                 else
                     flt_image = convert_image_dtype(image, dtypes.float32);

                 var adjusted = gen_ops.adjust_saturation(flt_image, saturation_factor);

                 return convert_image_dtype(adjusted, orig_dtype);
             });
        }

        public static Tensor total_variation(Tensor images, string name = null)
        {
            /*
            return tf_with(ops.name_scope(name, "total_variation"), delegate
            {
                
            });
            */
            throw new NotImplementedException("");
        }

        public static (Tensor begin, Tensor size, Tensor bboxes) sample_distorted_bounding_box_v2(Tensor image_size, Tensor bounding_boxes, int seed = 0,
            Tensor min_object_covered = null, float[] aspect_ratio_range = null, float[] area_range = null, int max_attempts = 100,
            bool use_image_if_no_bounding_boxes = false, string name = null)
        {
            // set default values that couldn't be set in function declaration, if necessary
            if (min_object_covered == null)
                min_object_covered = ops.convert_to_tensor(0.1);
            if (aspect_ratio_range == null)
                aspect_ratio_range = new float[] { 0.75f, 1.33f };
            if (area_range == null)
                area_range = new float[] { 0.05f, 1f };

            int? seed1, seed2;
            if (seed != 0)
                (seed1, seed2) = random_seed.get_seed(seed);
            else
                (seed1, seed2) = (0, 0);

            return sample_distorted_bounding_box(image_size, bounding_boxes, seed1, seed2,
                                                min_object_covered, aspect_ratio_range,
                                                area_range, max_attempts,
                                                use_image_if_no_bounding_boxes, name);
        }

        internal static (Tensor begin, Tensor size, Tensor bboxes) sample_distorted_bounding_box(Tensor image_size, Tensor bounding_boxes, int? seed = 0, int? seed2 = 0,
            Tensor min_object_covered = null, float[] aspect_ratio_range = null, float[] area_range = null, int max_attempts = 100,
            bool use_image_if_no_bounding_boxes = false, string name = null)
        {
            return tf_with(ops.name_scope(name, "sample_distorted_bounding_box"), delegate
            {
                return gen_ops.sample_distorted_bounding_box_v2(
                    image_size,
                    bounding_boxes,
                    seed: seed,
                    seed2: seed2,
                    min_object_covered: min_object_covered,
                    aspect_ratio_range: aspect_ratio_range,
                    area_range: area_range,
                    max_attempts: max_attempts,
                    use_image_if_no_bounding_boxes: use_image_if_no_bounding_boxes,
                    name: name);
            });
        }

        public static Tensor non_max_suppression(Tensor boxes, Tensor scores, Tensor max_output_size, float iou_threshold = 0.5f,
            float score_threshold = -1f / 0f, string name = null)
        {
            return tf_with(ops.name_scope(name, "non_max_suppression,"), delegate
            {
                Tensor iou_threshold_tensor = ops.convert_to_tensor(iou_threshold, name: "iou_threshold");
                Tensor score_threshold_tensor = ops.convert_to_tensor(score_threshold, name: "score_threshold");
                return gen_ops.non_max_suppression_v3(boxes, scores, max_output_size,
                                                    iou_threshold_tensor, score_threshold_tensor);
            });
        }

        public static (Tensor, Tensor) non_max_suppression_with_scores(Tensor boxes, Tensor scores, Tensor max_output_size,
            float iou_threshold = 0.5f, float score_threshold = -1f / 0f, /*float soft_nms_sigma = 0.0f,*/ string name = null)
        {
            return tf_with(ops.name_scope(name, "non_max_suppression_with_scores"), delegate
            {
                Tensor iou_threshold_tensor = ops.convert_to_tensor(iou_threshold, name: "iou_threshold");
                Tensor score_threshold_tensor = ops.convert_to_tensor(score_threshold, name: "score_threshold");

                // non_max_suppression_v5 apparently doesn't exist yet, so use v4
                // and adapt the arguments to fit

                // Tensor soft_nms_sigma_tensor = ops.convert_to_tensor(soft_nms_sigma, name: "soft_nms_sigma");
                (Tensor selected_indices, Tensor selected_scores) = gen_ops.non_max_suppression_v4(
                    boxes,
                    scores,
                    max_output_size,
                    iou_threshold_tensor,
                    score_threshold_tensor,
                    // soft_nms_sigma_tensor,
                    false
                );
                return (selected_indices, selected_scores);
            });
        }

        public static Tensor non_max_suppression_with_overlaps(Tensor overlaps, Tensor scores, Tensor max_output_size,
            float overlap_threshold = 0.5f, float score_threshold = -1f / 0f, string name = null)
        {
            return tf_with(ops.name_scope(name, "non_max_suppression_overlaps"), delegate
            {
                Tensor overlap_threshold_tensor = ops.convert_to_tensor(overlap_threshold, name: "overlap_threshold");
                return gen_ops.non_max_suppression_with_overlaps(
                    overlaps, scores, max_output_size, overlap_threshold_tensor, ops.convert_to_tensor(score_threshold));
            });
        }

        public static Tensor rgb_to_yiq(Tensor images)
        {
            images = ops.convert_to_tensor(images, name: "images");
            var _rgb_to_yiq_kernel = new float[,] { {0.299f, 0.59590059f, 0.2115f},
                                                    {0.587f, -0.27455667f, -0.52273617f},
                                                    {0.114f, -0.32134392f, 0.31119955f}};
            Tensor kernel = ops.convert_to_tensor(_rgb_to_yiq_kernel, dtype: images.dtype, name: "kernel");
            var ndims = images.TensorShape.ndim;
            return math_ops.tensordot(images, kernel, axes: new int[] { ndims - 1, 0 });
        }

        public static Tensor yiq_to_rgb(Tensor images)
        {
            images = ops.convert_to_tensor(images, name: "images");
            var _yiq_to_rgb_kernel = new float[,] { {1f, 1f, 1f},
                                                    {0.95598634f, -0.27201283f, -1.10674021f},
                                                    {0.6208248f, -0.64720424f, 1.70423049f}};
            Tensor kernel = ops.convert_to_tensor(_yiq_to_rgb_kernel, dtype: images.dtype, name: "kernel");
            var ndims = images.TensorShape.ndim;
            return math_ops.tensordot(images, kernel, axes: new int[] { ndims - 1, 0 });
        }

        public static Tensor rgb_to_yuv(Tensor images)
        {
            images = ops.convert_to_tensor(images, name: "images");
            var _rgb_to_yuv_kernel = new float[,] { {0.299f, -0.14714119f, 0.61497538f},
                                                    {0.587f, -0.28886916f, -0.51496512f},
                                                    {0.114f, 0.43601035f, -0.10001026f}};
            Tensor kernel = ops.convert_to_tensor(_rgb_to_yuv_kernel, dtype: images.dtype, name: "kernel");
            var ndims = images.TensorShape.ndim;
            return math_ops.tensordot(images, kernel, axes: new int[] { ndims - 1, 0 });
        }

        public static Tensor yuv_to_rgb(Tensor images)
        {
            images = ops.convert_to_tensor(images, name: "images");
            var _yuv_to_rgb_kernel = new float[,] { {1f, 1f, 1f,},
                                                    {0f, -0.394642334f, 2.03206185f},
                                                    {1.13988303f, -0.58062185f, 0f}};
            Tensor kernel = ops.convert_to_tensor(_yuv_to_rgb_kernel, dtype: images.dtype, name: "kernel");
            var ndims = images.TensorShape.ndim;
            return math_ops.tensordot(images, kernel, axes: new int[] { ndims - 1, 0 });
        }

        internal static (Tensor, Tensor, Operation[]) _verify_compatible_image_shapes(Tensor img1, Tensor img2)
        {
            TensorShape shape1 = img1.TensorShape.with_rank_at_least(3);
            TensorShape shape2 = img2.TensorShape.with_rank_at_least(3);
            shape1 = new TensorShape(shape1.dims.Skip(shape1.dims.Length - 3).Take(shape1.dims.Length - (shape1.dims.Length - 3)).ToArray());
            tensor_shape.assert_is_compatible_with(self: new Tensor(shape1), other: new Tensor(shape2.dims.Skip(shape2.dims.Length - 3).Take(shape2.dims.Length - (shape2.dims.Length - 3)).ToArray()));

            if (shape1.ndim != -1 && shape2.ndim != -1)
            {
                var shape1_temp = shape1.dims.Skip(shape1.dims.Length - 3).Take(shape1.dims.Length - (shape1.dims.Length - 3)).ToArray();
                var shape2_temp = shape2.dims.Skip(shape2.dims.Length - 3).Take(shape2.dims.Length - (shape1.dims.Length - 3)).ToArray();
                Array.Reverse(shape1_temp);
                Array.Reverse(shape2_temp);
                foreach ((int dim1, int dim2) in shape1_temp.Zip(shape2_temp, Tuple.Create))
                {
                    if (dim1 != 1 || dim2 != 1 /*|| !dim1.is_compatible_with(dim2)*/)
                        throw new ValueError(String.Format("Two images are not compatible: {0} and {1}", shape1, shape2));
                }
            }

            Tensor shape1_tensor = gen_array_ops.shape_n(new Tensor[] { img1, img2 })[0];
            Tensor shape2_tensor = gen_array_ops.shape_n(new Tensor[] { img1, img2 })[1];
            Operation[] checks = new Operation[] { };
            checks.append(
                control_flow_ops.Assert(
                    gen_math_ops.greater_equal(array_ops.size(shape1_tensor), 3), new[] { shape1, shape2 },
                    summarize: 10));
            checks.append(
                control_flow_ops.Assert(
                    math_ops.reduce_all(math_ops.equal(shape1_tensor.dims.Skip(shape1_tensor.dims.Length - 3).Take(shape1_tensor.dims.Length - (shape1_tensor.dims.Length - 3)).ToArray(),
                                        shape2_tensor.dims.Skip(shape1_tensor.dims.Length - 3).Take(shape1_tensor.dims.Length - (shape1_tensor.dims.Length - 3)))),
                                        new[] { shape1, shape2 },
                                        summarize: 10));
            return (shape1_tensor, shape2_tensor, checks);
        }

        public static Tensor psnr(Tensor a, Tensor b, Tensor max_val, string name = null)
        {
            return tf_with(ops.name_scope(name, "PSNR", new[] { a, b }), delegate
             {
                 max_val = math_ops.cast(max_val, a.dtype);
                 max_val = convert_image_dtype(max_val, dtypes.float32);
                 a = convert_image_dtype(a, dtypes.float32);
                 b = convert_image_dtype(b, dtypes.float32);
                 Tensor mse = math_ops.reduce_mean(gen_math_ops.squared_difference(a, b), new int[] { -3, -2, -1 });
                 var psnr_val = math_ops.subtract(
                     (20 * math_ops.log(max_val)) / math_ops.log(ops.convert_to_tensor(10.0)),
                     math_ops.cast(10 / math_ops.log(ops.convert_to_tensor(10)), dtypes.float32) * math_ops.log(mse),
                     name: "psnr");

                 (object _a, object _b, Operation[] checks) = _verify_compatible_image_shapes(a, b);
                 return tf_with(ops.control_dependencies(checks), delegate
                 {
                     return array_ops.identity(psnr_val);
                 });
             });
        }

        internal static (Tensor, Tensor) _ssim_helper(Tensor x, Tensor y, Func<Tensor, Tensor> reducer, float max_val,
            float compensation = 1.0f, float k1 = 0.01f, float k2 = 0.03f)
        {
            var c1 = Math.Pow((k1 * max_val), 2);
            var c2 = Math.Pow((k2 * max_val), 2);

            var mean0 = reducer(x);
            var mean1 = reducer(y);
            var num0 = mean0 * mean1 * 2.0;
            var den0 = math_ops.square(mean0) + math_ops.square(mean1);
            var luminance = (num0 + c1) / (den0 + c1);

            var num1 = reducer(x * y) * 2.0;
            var den1 = reducer(math_ops.square(x) + math_ops.square(y));
            c2 = c2 * compensation;
            var cs = (num1 - num0 + c2) / (den1 - den0 + c2);

            return (luminance, cs);
        }

        internal static Tensor _fspecial_gauss(Tensor size, Tensor sigma)
        {
            size = ops.convert_to_tensor(size, dtypes.int32);
            sigma = ops.convert_to_tensor(sigma);

            var coords = math_ops.cast(math_ops.range(size), sigma.dtype);
            coords = coords - math_ops.cast(size - 1, sigma.dtype) / 2.0;

            var g = math_ops.square(coords);
            g = g * -0.5 / math_ops.square(sigma);

            g = array_ops.reshape(g, shape: new int[] { 1, -1 }) + array_ops.reshape(g, shape: new int[] { -1, 1 });
            g = array_ops.reshape(g, shape: new int[] { 1, -1 });
            g = nn_ops.softmax(g);

            // shape takes an int, python code passes size, a Tensor. NDims is the only int type
            // i could think of a Tensor having. it might be incorrect tho, so keep that in mind.
            return array_ops.reshape(g, shape: new int[] { size.NDims, size.NDims, 1, 1 });
        }

        internal static (Tensor, Tensor) _ssim_per_channel(Tensor img1, Tensor img2, float max_val = 1f,
            float filter_size = 11f, float filter_sigma = 1.5f, float k1 = 0.01f, float k2 = 0.03f)
        {
            Tensor filter_size_tensor = constant_op.constant(filter_size, dtype: dtypes.int32);
            Tensor filter_sigma_tensor = constant_op.constant(filter_sigma, dtype: img1.dtype);

            Tensor shape1_tensor = gen_array_ops.shape_n(new Tensor[] { img1, img2 })[0];
            Tensor shape2_tensor = gen_array_ops.shape_n(new Tensor[] { img1, img2 })[1];
            Operation[] checks = new Operation[] {
                control_flow_ops.Assert(
                    math_ops.reduce_all(
                        gen_math_ops.greater_equal(new Tensor(shape1_tensor.dims.Skip(shape1_tensor.dims.Length - 3).Take(shape1_tensor.dims.Length - (shape1_tensor.dims.Length - 3 - 1)).ToArray()), filter_size_tensor)),
                    new object[] {shape1_tensor, filter_size},
                    summarize: 8),
                control_flow_ops.Assert(
                    math_ops.reduce_all(
                        gen_math_ops.greater_equal(new Tensor(shape2_tensor.dims.Skip(shape2_tensor.dims.Length - 3).Take(shape2_tensor.dims.Length - (shape2_tensor.dims.Length - 3 - 1)).ToArray()), filter_size_tensor)),
                    new object[] {shape2_tensor, filter_size},
                    summarize: 8)
            };

            using (ops.control_dependencies(checks))
                img1 = array_ops.identity(img1);

            var kernel = _fspecial_gauss(filter_size_tensor, filter_sigma_tensor);
            kernel = array_ops.tile(kernel, multiples: new Tensor(new int[] { 1, 1, shape1_tensor.dims[shape1_tensor.dims.Length - 2], 1 }));

            float compensation = 1.0f;

            Tensor reducer(Tensor x)
            {
                var shape = array_ops.shape(x);
                x = array_ops.reshape(x, shape: array_ops.concat(new Tensor[] { new Tensor(-1), new Tensor(shape1_tensor.dims.Skip(shape1_tensor.dims.Length - 3).Take(shape1_tensor.dims.Length - (shape1_tensor.dims.Length - 3 - 1)).ToArray()) }, 0));
                var y = gen_ops.depthwise_conv2d_native(x, kernel, strides: new int[] { 1, 1, 1, 1 }, padding: "VALID");
                return array_ops.reshape(
                    y, array_ops.concat(new Tensor[] { new Tensor(shape.dims.Take(shape.dims.Length - 3).ToArray()), new Tensor(array_ops.shape(y).dims.Skip(1).Take(array_ops.shape(y).dims.Length - 2).ToArray()) }, 0));
            }

            (Tensor luminance, Tensor cs) = _ssim_helper(img1, img2, reducer, max_val, compensation, k1, k2);

            var axes = constant_op.constant(new[] { -3, -2 }, dtype: dtypes.int32);
            var ssim_val = math_ops.reduce_mean(luminance * cs, axes.dims);
            cs = math_ops.reduce_mean(cs, axes.dims);
            return (ssim_val, cs);
        }

        public static Tensor ssim(Tensor img1, Tensor img2, float max_val = 1f, float filter_size = 11f, float filter_sigma = 1.5f,
            float k1 = 0.01f, float k2 = 0.03f)
        {
            return tf_with(ops.name_scope(null, "SSIM", new[] { img1, img2 }), delegate
             {
                 img1 = ops.convert_to_tensor(img1, name: "img1");
                 img2 = ops.convert_to_tensor(img2, name: "img2");

                 (Tensor _, Tensor __, Operation[] checks) = _verify_compatible_image_shapes(img1, img2);
                 using (ops.control_dependencies(checks))
                     img1 = array_ops.identity(img1);

                 Tensor max_val_tensor = math_ops.cast(max_val, img1.dtype);
                 max_val_tensor = convert_image_dtype(max_val_tensor, dtypes.float32);
                 img1 = convert_image_dtype(img1, dtypes.float32);
                 img2 = convert_image_dtype(img2, dtypes.float32);
                 (Tensor ssim_per_channel, Tensor ___) = _ssim_per_channel(img1, img2, max_val, filter_size,
                                                                             filter_sigma, k1, k2);

                 return math_ops.reduce_mean(ssim_per_channel, new int[] { -1 });
             });
        }

        public static Tensor ssim_multiscale(Tensor img1, Tensor img2, float max_val, float[] power_factors = null, float filter_size = 11f,
            float filter_sigma = 1.5f, float k1 = 0.01f, float k2 = 0.03f)
        {
            if (power_factors == null)
                power_factors = new float[] { 0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f };

            return tf_with(ops.name_scope(null, "MS-SSIM", new[] { img1, img2 }), delegate
             {
                 img1 = ops.convert_to_tensor(img1, name: "img1");
                 img2 = ops.convert_to_tensor(img2, name: "img2");

                 (Tensor shape1, Tensor shape2, Operation[] checks) = _verify_compatible_image_shapes(img1, img2);
                 using (ops.control_dependencies(checks))
                     img1 = array_ops.identity(img1);

                 Tensor max_val_tensor = math_ops.cast(max_val, img1.dtype);
                 max_val_tensor = convert_image_dtype(max_val_tensor, dtypes.float32);
                 img1 = convert_image_dtype(img1, dtypes.float32);
                 img2 = convert_image_dtype(img2, dtypes.float32);

                 var imgs = new[] { img1, img2 };
                 var shapes = new[] { shape1, shape2 };

                 Tensor[] heads = new Tensor[] { };
                 Tensor[] tails = new Tensor[] { };
                 foreach (Tensor s in shapes)
                 {
                     heads[heads.Length] = new Tensor(s.dims.Take(s.dims.Length - 3).ToArray());
                     tails[tails.Length] = new Tensor(s.dims.Skip(s.dims.Length - 3).Take(s.dims.Length - (s.dims.Length - 3)).ToArray());
                 }

                 var divisor = new[] { 1, 2, 2, 1 };
                 var divisor_tensor = constant_op.constant(divisor.Skip(1).Take(divisor.Length - 1).ToArray(), dtype: dtypes.int32);

                 Tensor[] do_pad(Tensor[] images, Tensor remainder)
                 {
                     var padding = array_ops.expand_dims(remainder, -1);
                     padding = array_ops.pad(padding, new Tensor(new int[,] { { 1, 0 }, { 1, 0 } }));

                     Tensor[] x_arr = new Tensor[] { };
                     foreach (Tensor x in images)
                     {
                         x_arr[x_arr.Length] = array_ops.pad(x, padding, mode: "SYMMETRIC");
                     }
                     return x_arr;
                 }

                 var mcs = new Tensor[] { };
                 var ssim_per_channel = new Tensor(new int[] { });
                 var cs = ssim_per_channel;
                 foreach (var k in range(0, len(power_factors)))
                 {
                     using (ops.name_scope(null, String.Format("Scale{0}", k), imgs))
                     {
                         if (k > 0)
                         {
                             // handle flat_imgs
                             Tensor[] flat_imgs = new Tensor[] { };
                             foreach ((Tensor x, Tensor t) in imgs.Zip(tails, Tuple.Create))
                             {
                                 flat_imgs[flat_imgs.Length] = array_ops.reshape(x, array_ops.concat(new Tensor[] { constant_op.constant(-1), t }, 0));
                             }

                             var remainder = tails[0] % divisor_tensor;
                             var need_padding = math_ops.reduce_any(math_ops.not_equal(remainder, 0));

                             Tensor[] padded_func_pass() { return do_pad(flat_imgs, remainder); }
                             var padded = control_flow_ops.cond(need_padding,
                                                                 true_fn: () => padded_func_pass(),
                                                                 false_fn: () => flat_imgs);

                             // handle downscaled
                             Tensor[] downscaled = new Tensor[] { };
                             foreach (Tensor x in padded)
                             {
                                 downscaled[downscaled.Length] = gen_ops.avg_pool(x, ksize: divisor, strides: divisor, padding: "VALID");
                             }

                             // handle tails
                             tails = new Tensor[] { };
                             foreach (Tensor x in gen_array_ops.shape_n(downscaled))
                             {
                                 tails[tails.Length] = new Tensor(x.dims.Skip(1).Take(tails.Length - 1).ToArray());
                             }

                             imgs = new Tensor[] { };
                             // tuples weren't working; this is hacky, but should work similarly.
                             // zip loads the values into a tuple (Tensor, Tensor, Tensor) for each
                             // zip entry; this just gets the length of the longest array, and loops
                             // that many times, getting values (like zip) and using them similarly.
                             for (int x = 0; x < Math.Max(Math.Max(downscaled.Length, heads.Length), tails.Length); x++)
                             {
                                 imgs[imgs.Length] = array_ops.reshape(downscaled[x], array_ops.concat(new Tensor[] { heads[x], tails[x] }, 0));
                             }
                         }
                     }

                     // python code uses * to unpack imgs; how to replicate that here?
                     // don't think that this is doing the same thing as the python code.
                     (ssim_per_channel, cs) = _ssim_per_channel(
                          img1: imgs[0],
                          img2: imgs[1],
                          max_val: max_val,
                          filter_size: filter_size,
                          filter_sigma: filter_sigma,
                          k1: k1,
                          k2: k2);
                     mcs.append(gen_nn_ops.relu(cs));
                 }

                 mcs = mcs.Skip(1).ToArray();
                 var mcs_and_ssim = array_ops.stack(
                     math_ops.add(mcs, new[] { gen_nn_ops.relu(ssim_per_channel) }), axis: -1);
                 var ms_ssim = math_ops.reduce_prod(
                     math_ops.pow(mcs_and_ssim, power_factors), new int[] { -1 });

                 return math_ops.reduce_mean(ms_ssim, new int[] { -1 });
             });
        }

        public static (Tensor, Tensor) image_gradients(Tensor image)
        {
            if (image.TensorShape.ndim != 4)
                throw new ValueError(String.Format(@"image_gradients expects a 4D tensor [batch_size, h, w, d], not {0}.", image.shape));

            var image_shape = array_ops.shape(image);
            var bs_h_w_d = array_ops.unstack(image_shape);
            Tensor dy; //= image[:, 1:, :, :] - image[:, :-1, :, :];
            Tensor dx = new Tensor(new int[] { }); //= image[:, :, 1:, :] - image[:, :, :-1, :];

            var shape = array_ops.stack(new Tensor[] { bs_h_w_d[0], constant_op.constant(1), bs_h_w_d[2], bs_h_w_d[3] });
            dy = array_ops.concat(new Tensor[] { dx, array_ops.zeros(shape, image.dtype) }, 2);
            dy = array_ops.reshape(dy, image_shape);

            shape = array_ops.stack(new Tensor[] { bs_h_w_d[0], bs_h_w_d[1], constant_op.constant(1), bs_h_w_d[3] });
            dx = array_ops.concat(new Tensor[] { dx, array_ops.zeros(shape, image.dtype) }, 2);
            dx = array_ops.reshape(dx, image_shape);

            return (dx, dy);
        }

        public static Tensor sobel_edges(Tensor image)
        {
            var static_image_shape = image.TensorShape;
            var image_shape = array_ops.shape(image);
            var kernels = new Tensor(new int[,] {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1},
                                                 {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}});
            var num_kernels = len(kernels);
            // kernels.dims != np.asarray(kernels) ?
            kernels = array_ops.transpose(kernels.dims, (1, 2, 0));
            kernels = array_ops.expand_dims(kernels, -2);
            var kernels_tf = constant_op.constant(kernels, dtype: image.dtype);

            kernels_tf = array_ops.tile(
                kernels_tf, new Tensor(new int[] { 1, 1, image_shape.dims[image_shape.dims.Length - 2], 1 }), name: "sobel_filters");

            var pad_sizes = new int[,] { { 0, 0 }, { 1, 1 }, { 1, 1 }, { 0, 0 } };
            var padded = array_ops.pad(image, new Tensor(pad_sizes), mode: "reflect");

            var strides = new int[] { 1, 1, 1, 1 };
            var output = gen_ops.depthwise_conv2d_native(padded, kernels_tf, strides, "VALID");

            var shape = array_ops.concat(new Tensor[] { image_shape, ops.convert_to_tensor(num_kernels) }, 0);
            output = array_ops.reshape(output, shape: shape);
            output.set_shape(static_image_shape.concatenate(new int[] { num_kernels }));
            return output;
        }

        public static Tensor decode_image(Tensor contents, int channels = 0, TF_DataType dtype = TF_DataType.TF_UINT8,
            string name = null, bool expand_animations = true)
        {
            return tf_with(ops.name_scope(name, "decode_image"), scope =>
            {
                var substr = tf.strings.substr(contents, 0, 3);

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

                /*Func<ITensorOrOperation> _gif = () =>
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
                            result = array_ops.gather(result, 0);
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
                    var gif = tf.constant(new byte[] { 0x47, 0x49, 0x46 }, TF_DataType.TF_STRING);
                    var is_gif = math_ops.equal(substr, gif, name: name);
                    return control_flow_ops.cond(is_gif, _gif, _bmp, name: "cond_gif");
                };

                Func<ITensorOrOperation> check_png = () =>
                {
                    return control_flow_ops.cond(is_png(contents), _png, check_gif, name: "cond_png");
                };*/

                // return control_flow_ops.cond(is_jpeg(contents), _jpeg, check_png, name: "cond_jpeg");
                return _jpeg() as Tensor;
            });
        }

        public static Tensor crop_and_resize(Tensor image, Tensor boxes, Tensor box_ind, Tensor crop_size, string method, float extrapolation_value, string name)
        {
            var _op = tf.OpDefLib._apply_op_helper("CropAndResize", name: name, args: new
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

        public static Tensor extract_glimpse(Tensor input, Tensor size, Tensor offsets, bool centered = true, bool normalized = true,
            bool uniform_noise = true, string name = null)
        {
            return gen_ops.extract_glimpse(
                input: input,
                size: size,
                offsets: offsets,
                centered: centered,
                normalized: normalized,
                uniform_noise: uniform_noise,
                name: name);
        }

        public static (Tensor, Tensor, Tensor, Tensor) combined_non_max_suppression(Tensor boxes, Tensor scores, Tensor max_output_size_per_class,
            Tensor max_total_size, float iou_threshold = 0.5f, float score_threshold = -1f / 0f, bool pad_per_class = false, bool clip_boxes = true,
            string name = null)
        {
            return tf_with(ops.name_scope(null, "combined_non_max_suppression"), delegate
            {
                Tensor iou_threshold_tensor = ops.convert_to_tensor(
                    iou_threshold, dtype: dtypes.float32, name: "iou_threshold");
                Tensor score_threshold_tensor = ops.convert_to_tensor(
                    score_threshold, dtype: dtypes.float32, name: "score_threshold");
                return gen_image_ops.combined_non_max_suppression(
                    boxes, scores, max_output_size_per_class, max_total_size, iou_threshold_tensor,
                    score_threshold_tensor, pad_per_class, clip_boxes);
            });
        }

        internal static (Tensor, Tensor, Tensor, Tensor) _cross_suppression(Tensor boxes, Tensor box_slice, Tensor iou_threshold, Tensor inner_idx, int tile_size)
        {
            var batch_size = array_ops.shape(boxes)[0];
            var new_slice = array_ops.slice(
                boxes, new object[] { 0, inner_idx * tile_size, 0 },
                new object[] { batch_size, tile_size, 4 });
            var iou = _bbox_overlap(new_slice, box_slice);
            var box_slice_after_suppression = array_ops.expand_dims(
                math_ops.cast(math_ops.reduce_all(iou < iou_threshold, new int[] { 1 }),
                                box_slice.dtype),
                2) * box_slice;
            return (boxes, box_slice_after_suppression, iou_threshold, inner_idx + 1);
        }

        internal static Tensor _bbox_overlap(Tensor boxes_a, Tensor boxes_b)
        {
            return tf_with(ops.name_scope("bbox_overlap"), delegate
            {
                // a_y_min: [0], a_x_min: [1], a_y_max: [2], a_x_max[3]
                var a_xy_minmax = array_ops.split(
                    value: boxes_a, num_split: 4, axis: 2);
                // b_y_min: [0], b_x_min: [1], b_y_max: [2], b_x_max[3]    
                var b_xy_minmax = array_ops.split(
                    value: boxes_b, num_split: 4, axis: 2);

                var i_xmin = math_ops.maximum(
                    a_xy_minmax[1], array_ops.transpose(b_xy_minmax[1], new[] { 0, 2, 1 }));
                var i_xmax = math_ops.minimum(
                    a_xy_minmax[3], array_ops.transpose(b_xy_minmax[3], new[] { 0, 2, 1 }));
                var i_ymin = math_ops.maximum(
                    a_xy_minmax[0], array_ops.transpose(b_xy_minmax[0], new[] { 0, 2, 1 }));
                var i_ymax = math_ops.minimum(
                    a_xy_minmax[3], array_ops.transpose(b_xy_minmax[3], new[] { 0, 2, 1 }));
                var i_area = math_ops.maximum(
                    (i_xmax - i_xmin), 0) * math_ops.maximum((i_ymax - i_ymin), 0);

                var a_area = (a_xy_minmax[2] - a_xy_minmax[0]) * (a_xy_minmax[3] - a_xy_minmax[1]);
                var b_area = (b_xy_minmax[2] - b_xy_minmax[0]) * (b_xy_minmax[3] - b_xy_minmax[1]);
                double EPSILON = 1e-8;

                var u_area = a_area + array_ops.transpose(b_area, new[] { 0, 2, 1 }) - i_area + EPSILON;

                var intersection_over_union = i_area / u_area;

                return intersection_over_union;
            });
        }

        internal static (Tensor, float, Tensor, int) _suppression_loop_body(Tensor boxes, float iou_threshold, Tensor output_size, int idx, int tile_size)
        {
            using (ops.name_scope("suppression_loop_body"))
            {
                var num_tiles = array_ops.shape(boxes).dims[1] / tile_size;
                var batch_size = array_ops.shape(boxes).dims[0];

                (Tensor, Tensor, Tensor, Tensor) cross_suppression_func(Tensor boxes, Tensor box_slice, Tensor iou_threshold, Tensor inner_idx, int tile_size)
                    => _cross_suppression(boxes, box_slice, iou_threshold, inner_idx, tile_size);

                var box_slice = array_ops.slice(boxes, new[] { 0, idx * tile_size, 0 },
                                                new[] { batch_size, tile_size, 4 });

                var iou = _bbox_overlap(box_slice, box_slice);
                var mask = array_ops.expand_dims(
                    array_ops.reshape(
                        math_ops.range(tile_size), new[] { 1, -1 }) > array_ops.reshape(
                            math_ops.range(tile_size), new[] { -1, 1 }), 0);
                iou = iou * math_ops.cast(
                    math_ops.logical_and(mask, iou >= iou_threshold), iou.dtype);

                /*
                I have no idea what's going on here. Not even going to try to port it yet.
                var suppressed_iou = control_flow_ops.while_loop(
                    todo
                )
                */
                var suppressed_iou = new Tensor(new int[] { });
                var suppressed_box = math_ops.reduce_sum(suppressed_iou, 1) > 0;
                box_slice = box_slice * array_ops.expand_dims(
                    1.0f - math_ops.cast(suppressed_box, box_slice.dtype), 2);

                mask = array_ops.reshape(
                    math_ops.cast(
                        math_ops.equal(math_ops.range(num_tiles), idx), boxes.dtype),
                    new[] { 1, -1, 1, 1 });
                boxes = array_ops.tile(array_ops.expand_dims(
                    box_slice, 1), ops.convert_to_tensor(new[] { 1, num_tiles, 1, 1 }) * mask + array_ops.reshape(
                        boxes, new[] { batch_size, num_tiles, tile_size, 4 }) * (1 - mask));
                boxes = array_ops.reshape(boxes, new[] { batch_size, -1, 4 });

                output_size = output_size + math_ops.reduce_sum(
                    math_ops.cast(
                        math_ops.reduce_any(box_slice > 0, new int[] { 2 }), dtypes.int32), new int[] { 1 });
            }
            return (boxes, iou_threshold, output_size, idx + 1);
        }

        public static (Tensor, Tensor) non_max_suppression_padded(Tensor boxes, Tensor scores, Tensor max_output_size, float iou_threshold = 0.5f, float score_threshold = -1f / 0f,
            bool pad_to_max_output_size = false, string name = null, bool sorted_input = false, bool canonicalized_coordinates = false, int tile_size = 512)
        {
            if (!sorted_input && !canonicalized_coordinates && tile_size == 512 /*&& !compat.forward_compatible(2020, 6, 23)*/)
                return non_max_suppression_padded_v1(
                    boxes, scores, max_output_size, iou_threshold, score_threshold,
                    pad_to_max_output_size, name);
            else
            {
                return tf_with(ops.name_scope(name, "non_max_suppression_padded"), delegate
                {
                    if (!pad_to_max_output_size)
                        if (boxes.TensorShape.rank != -1 && boxes.TensorShape.rank > 2)
                            throw new ValueError(String.Format(
                                "'pad_to_max_output_size' (value {0}) must be true for 'batched input'", pad_to_max_output_size));
                    if (name == null)
                        name = "";
                    (Tensor idx, Tensor num_valid) = non_max_suppression_padded_v2(
                        boxes, scores, max_output_size, iou_threshold, score_threshold,
                        sorted_input, canonicalized_coordinates, tile_size);
                    if (!pad_to_max_output_size)
                        // idx = idx[0, :num_valid], passes:
                        //   0, slice(None, num_valid, None)
                        // which is what I tried to replicate below, but i don't think that Unknown is the exact
                        // equivalent to None, and don't know about the slice function bit.
                        idx = idx[0, slice(Unknown, num_valid.TensorShape.ndim, Unknown).ToArray()[0]];
                    else
                    {
                        var batch_dims = array_ops.concat(new Tensor[] {
                            new Tensor(array_ops.shape(boxes).dims.Take(boxes.TensorShape.dims.Length - 2).ToArray()),
                            array_ops.expand_dims(max_output_size, 0)
                        }, 0);
                        idx = array_ops.reshape(idx, batch_dims);
                    }
                    return (idx, num_valid);
                });
            }
        }

        public static (Tensor, Tensor) non_max_suppression_padded_v2(Tensor boxes, Tensor scores, Tensor max_output_size, float iou_threshold = 0.5f, float score_threshold = -1f / 0f,
            bool sorted_input = false, bool canonicalized_coordinates = false, int tile_size = 512)
        {
            (Tensor, Tensor, Tensor) _sort_scores_and_boxes(Tensor scores, Tensor boxes)
            {
                int batch_size, num_boxes;
                Tensor index_offsets, indices, sorted_scores, sorted_boxes, sorted_scores_indices;
                using (ops.name_scope("sort_scores_and_boxes"))
                {
                    batch_size = array_ops.shape(boxes).dims[0];
                    num_boxes = array_ops.shape(boxes).dims[1];
                    sorted_scores_indices = null; /*sort_ops.argsort(
                        scores, axis: 1, direction: "DESCENDING); */
                    index_offsets = math_ops.range(batch_size) * num_boxes;
                    indices = array_ops.reshape(
                        sorted_scores_indices + array_ops.expand_dims(index_offsets, 1), new[] { -1 });
                    sorted_scores = array_ops.reshape(
                        array_ops.gather(array_ops.reshape(boxes, new[] { -1, 4 }), indices),
                        new[] { batch_size, -1 });
                    sorted_boxes = array_ops.reshape(
                        array_ops.gather(array_ops.reshape(boxes, new[] { -1, 4 }), indices),
                        new[] { batch_size, -1, 4 });
                };

                return (sorted_scores, sorted_boxes, sorted_scores_indices);
            }

            var batch_dims = array_ops.shape(boxes).dims.Take(boxes.TensorShape.dims.Length - 2).ToArray();
            var num_boxes = array_ops.shape(boxes).dims[boxes.TensorShape.dims.Length - 2];
            boxes = array_ops.reshape(boxes, new[] { -1, num_boxes, 4 });
            scores = array_ops.reshape(scores, new[] { -1, num_boxes });
            var batch_size = array_ops.shape(boxes).dims[0];

            // initialization for later
            Tensor sorted_indices;

            if (score_threshold != -1f / 0f)
                using (ops.name_scope("filter_by_score"))
                {
                    var score_mask = math_ops.cast(scores > score_threshold, scores.dtype);
                    scores = scores * score_mask;
                    var box_mask = array_ops.expand_dims(
                        math_ops.cast(score_mask, boxes.dtype), 2);
                    boxes = boxes * box_mask;
                }

            if (!canonicalized_coordinates)
                using (ops.name_scope("canonicalize_coordinates"))
                {
                    // y_1 = [0], x_1 = [1], y_2 = [2], x_2 = [3]
                    var yx = array_ops.split(value: boxes, num_split: 4, axis: 2);
                    var y_1_is_min = math_ops.reduce_all(
                        gen_math_ops.less_equal(yx[0][0, 0, 0], yx[2][0, 0, 0]));
                    var y_minmax = control_flow_ops.cond(
                        y_1_is_min, true_fn: () => yx[0] /*yx[2]*/, false_fn: () => yx[2] /*yx[0]*/);
                    var x_1_is_min = math_ops.reduce_all(
                        gen_math_ops.less_equal(yx[1][0, 0, 0], yx[3][0, 0, 0]));
                    var x_minmax = control_flow_ops.cond(
                        x_1_is_min, true_fn: () => yx[1] /*yx[3]*/, false_fn: () => yx[3] /*yx[1]*/);
                    boxes = array_ops.concat(new Tensor[] { y_minmax, x_minmax }, axis: 2);
                }

            if (!sorted_input)
                (scores, boxes, sorted_indices) = _sort_scores_and_boxes(scores, boxes);
            else
                sorted_indices = array_ops.zeros_like(scores, dtype: dtypes.int32);

            var pad = math_ops.cast(
                gen_math_ops.ceil(
                    math_ops.cast(
                        math_ops.maximum(num_boxes, max_output_size), dtypes.float32) /
                    math_ops.cast(tile_size, dtypes.float32)),
                dtypes.int32) * tile_size - num_boxes;
            boxes = array_ops.pad(
                math_ops.cast(scores, dtypes.float32), ops.convert_to_tensor(new object[,] { { 0, 0 }, { 0, pad }, { 0, 0 } }));
            scores = array_ops.pad(
                math_ops.cast(scores, dtypes.float32), ops.convert_to_tensor(new object[,] { { 0, 0 }, { 0, pad } }));
            var num_boxes_after_padding = num_boxes + pad;
            var num_iterations = math_ops.floordiv(num_boxes_after_padding, ops.convert_to_tensor(tile_size));

            // Tensor unused_boxes, Tensor unused_threshold, Tensor output_size, Tensor idx go into args
            Tensor _loop_cond(object[] args)
                => /*new object[] {*/math_ops.logical_and(
                    math_ops.reduce_min((Tensor)args[2]) < max_output_size,
                    (Tensor)args[3] < num_iterations);

            // Tensor boxes, Tensor iou_threshold, Tensor output_size, Tensor idx go into args
            object[] suppression_loop_body(object[] args)
            {
                (Tensor a, float b, Tensor c, int d) = _suppression_loop_body((Tensor)args[0], (float)args[1], (Tensor)args[2], (int)args[3], tile_size);
                return new object[] { a, b, c, d };
            }

            object[] selboxes__output_size_ = null;
            /*
            errors here regarding the while loop and types

            object[] selboxes__output_size_= control_flow_ops.while_loop(
                cond: (Tensor[] args) => _loop_cond(args),
                body: (Tensor[] args) => suppression_loop_body(args),
                loop_vars: new object[] {
                    boxes, iou_threshold,
                    array_ops.zeros(new TensorShape(batch_size), dtypes.int32),
                    constant_op.constant(0)
                },
                shape_invariants: new TensorShape[] {
                    new TensorShape(new int[] {Unknown, Unknown, 4}),
                    new TensorShape(new int[] {}),
                    new TensorShape(new int[] {Unknown}),
                    new TensorShape(new int[] {})
                }
            );
            */
            var num_valid = math_ops.minimum(selboxes__output_size_[2], max_output_size);

            (Tensor values, Tensor indices) = gen_ops.top_k_v2(
                                                math_ops.cast(math_ops.reduce_any(
                                                    (Tensor)selboxes__output_size_[0] > 0, new int[] { 2 }), dtypes.int32) *
                                                array_ops.expand_dims(
                                                    math_ops.range(num_boxes_after_padding, 0, -1), 0),
                                                max_output_size);
            Tensor idx = num_boxes_after_padding - math_ops.cast(values.dims[0], dtypes.int32);
            idx = math_ops.minimum(idx, num_boxes - 1);

            if (!sorted_input)
            {
                var index_offsets = math_ops.range(batch_size) * num_boxes;
                var gather_idx = array_ops.reshape(
                    idx + array_ops.expand_dims(index_offsets, 1), new[] { -1 });
                idx = array_ops.reshape(
                    array_ops.gather(array_ops.reshape(sorted_indices, new[] { -1 }),
                                    gather_idx),
                    new[] { batch_size, -1 });
            }
            var invalid_index = array_ops.fill(ops.convert_to_tensor(new object[] { batch_size, max_output_size }),
                                                                    tf.constant(0));
            var idx_index = array_ops.expand_dims(math_ops.range(max_output_size), 0);
            var num_valid_expanded = array_ops.expand_dims(num_valid, 1);
            idx = array_ops.where(idx_index < num_valid_expanded,
                                    idx, invalid_index);
            num_valid = array_ops.reshape(num_valid, batch_dims);
            return (idx, num_valid);
        }

        internal static (Tensor, Tensor) non_max_suppression_padded_v1(Tensor boxes, Tensor scores, Tensor max_output_size, float iou_threshold = 0.5f,
            float score_threshold = -1f / 0f, bool pad_to_max_output_size = false, string name = null)
        {
            return tf_with(ops.name_scope(name, "non_max_supression_padded"), delegate
            {
                var iou_threshold_tensor = ops.convert_to_tensor(iou_threshold, name: "iou_threshold");
                var score_threshold_tensor = ops.convert_to_tensor(score_threshold, name: "score_threshold");
                return gen_ops.non_max_suppression_v4(boxes, scores, max_output_size, iou_threshold_tensor, score_threshold_tensor, pad_to_max_output_size);
            });
        }

        public static Tensor is_jpeg(Tensor contents, string name = null)
        {
            return tf_with(ops.name_scope(name, "is_jpeg"), scope =>
            {
                var substr = tf.strings.substr(contents, 0, 3);
                var jpg = tf.constant(new byte[] { 0xff, 0xd8, 0xff }, TF_DataType.TF_STRING);
                var result = math_ops.equal(substr, jpg, name: name);
                return result;
            });
        }

        static Tensor is_png(Tensor contents, string name = null)
        {
            return tf_with(ops.name_scope(name, "is_png"), scope =>
            {
                var substr = tf.strings.substr(contents, 0, 3);
                return math_ops.equal(substr, @"\211PN", name: name);
            });
        }

        static Tensor is_gif(Tensor contents, string name = null)
        {
            return tf_with(ops.name_scope(name, "is_gif"), scope =>
            {
                var substr = tf.strings.substr(contents, 0, 3);
                var gif = tf.constant(new byte[] { 0x47, 0x49, 0x46 }, TF_DataType.TF_STRING);
                var result = math_ops.equal(substr, gif, name: name);
                return result;
            });
        }

        public static Tensor convert_image_dtype(Tensor image, TF_DataType dtype, bool saturate = false,
            string name = null)
        {
            image = ops.convert_to_tensor(image, name: "image");
            // var tf_dtype = dtypes.as_dtype(dtype);
            if (!dtype.is_floating() && !dtype.is_integer())
                throw new TypeError("dtype must be either floating point or integer");
            if (dtype == image.dtype)
                return array_ops.identity(image, name: name);

            // declarations for later
            Tensor cast;

            return tf_with(ops.name_scope(name, "convert_image", new[] { image }), name =>
             {
                 if (image.dtype.is_integer() && dtype.is_integer())
                 {
                     var scale_in = image.dtype.max();
                     var scale_out = dtype.max();
                     if (scale_in > scale_out)
                     {
                         var scale = Math.Floor((decimal)(scale_in + 1) / (scale_out + 1));
                         var scaled = math_ops.floordiv(image, ops.convert_to_tensor(scale));

                         if (saturate)
                             return math_ops.saturate_cast(scaled, dtype, name: name);
                         else
                             return math_ops.cast(scaled, dtype, name: name);
                     }
                     else
                     {
                         if (saturate)
                             cast = math_ops.saturate_cast(image, dtype);
                         else
                             cast = math_ops.cast(image, dtype);
                         var scale = Math.Floor((decimal)(scale_in + 1) / (scale_out + 1));
                         return math_ops.multiply(cast, scale, name: name);
                     }
                 }
                 else if (image.dtype.is_floating() && dtype.is_floating())
                     return math_ops.cast(image, dtype, name: name);
                 else
                 {
                     if (image.dtype.is_integer())
                     {
                         cast = math_ops.cast(image, dtype);
                         var scale = 1 / image.dtype.max();
                         return math_ops.multiply(cast, scale, name: name);
                     }
                     else
                     {
                         var scale = dtype.max() + 0.5;
                         var scaled = math_ops.multiply(image, scale);
                         if (saturate)
                             return math_ops.saturate_cast(scaled, dtype, name: name);
                         else
                             return math_ops.cast(scaled, dtype, name: name);
                     }
                 }
             });
        }

        /// <summary>
        /// Resize `images` to `size` using the specified `method`.
        /// </summary>
        /// <param name="images"></param>
        /// <param name="size"></param>
        /// <param name="method"></param>
        /// <param name="preserve_aspect_ratio"></param>
        /// <param name="antialias"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor resize_images_v2<T>(Tensor images, T size, string method = ResizeMethod.BILINEAR,
            bool preserve_aspect_ratio = false,
            bool antialias = false,
            string name = null)
        {
            Func<Tensor, Tensor, Tensor> resize_fn = (images, size) =>
            {
                if (method == ResizeMethod.BILINEAR)
                    return gen_image_ops.resize_bilinear(images, size, half_pixel_centers: true);
                else if (method == ResizeMethod.NEAREST_NEIGHBOR)
                    return gen_image_ops.resize_nearest_neighbor(images, size, half_pixel_centers: true);

                throw new NotImplementedException("resize_images_v2");
            };
            return _resize_images_common(images, resize_fn, ops.convert_to_tensor(size),
                preserve_aspect_ratio: preserve_aspect_ratio,
                skip_resize_if_same: false,
                name: name);
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

        public static Tensor draw_bounding_boxes(Tensor images, Tensor boxes, Tensor colors = null, string name = null)
        {
            if (colors == null)
                return gen_ops.draw_bounding_boxes(images, boxes, name);
            return gen_ops.draw_bounding_boxes(images, boxes, /*colors,*/ name);
        }

        // TOOD: implement arguments, gen_ops
        public static Tensor generate_bounding_box_proposals()
        {
            throw new NotImplementedException("generate_bounding_box_propsosals");
        }
    }

    public class ResizeMethod
    {
        public ResizeMethod()
        {
        }

        public const string BILINEAR = "bilinear";
        public const string NEAREST_NEIGHBOR = "nearest";
        public const string BICUBIC = "bicubic";
        public const string AREA = "area";
        public const string LANCZOS3 = "lanczos3";
        public const string LANCZOS5 = "lanczos5";
        public const string GAUSSIAN = "gaussian";
        public const string MITCHELLCUBIC = "mitchellcubic";
    }
}
