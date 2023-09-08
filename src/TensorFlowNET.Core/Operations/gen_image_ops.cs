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

using System;
using System.Linq;
using Tensorflow.Eager;
using static Tensorflow.Binding;
using Tensorflow.Exceptions;
using Tensorflow.Contexts;
using System.Xml.Linq;
using Google.Protobuf;

namespace Tensorflow
{
    public class gen_image_ops
    {
        public static Tensor adjust_contrastv2(Tensor images, Tensor contrast_factor, string name = null)
        {
            var _ctx = tf.Context;
            if (_ctx.executing_eagerly())
            {
                try
                {
                    var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "AdjustContrastv2", name) { 
                                                args = new object[] { images, contrast_factor }, attrs = new Dictionary<string, object>() { } });
                    return _fast_path_result[0];
                }
                catch (NotOkStatusException ex)
                {
                    throw ex;
                }
                catch (Exception)
                {
                }
                try
                {
                    return adjust_contrastv2_eager_fallback(images, contrast_factor, name: name, ctx: _ctx);
                }
                catch (Exception)
                {
                }
            }
            Dictionary<string, object> keywords = new();
            keywords["images"] = images;
            keywords["contrast_factor"] = contrast_factor;
            var _op = tf.OpDefLib._apply_op_helper("AdjustContrastv2", name, keywords);
            var _result = _op.outputs;
            if (_execute.must_record_gradient())
            {
                object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
                _execute.record_gradient("AdjustContrastv2", _op.inputs, _attrs, _result);
            }
            return _result[0];
        }
        public static Tensor adjust_contrastv2(Tensor image, float contrast_factor, string name = null)
        {
            return adjust_contrastv2(image, tf.convert_to_tensor(contrast_factor), name: name);
        }

        public static Tensor adjust_contrastv2_eager_fallback(Tensor images, Tensor contrast_factor, string name, Context ctx)
        {
            Tensor[] _inputs_flat = new Tensor[] { images, contrast_factor};
            object[] _attrs = new object[] { "T", images.dtype };
            var _result = _execute.execute("AdjustContrastv2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
            if (_execute.must_record_gradient())
            {
                _execute.record_gradient("AdjustContrastv2", _inputs_flat, _attrs, _result);
            }
            return _result[0];
        }

        public static Tensor adjust_hue(Tensor images, Tensor delta, string name = null)
        {
            var _ctx = tf.Context;
            if (_ctx.executing_eagerly())
            {
                try
                {
                    var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "AdjustHue", name) { 
                                                args = new object[] { images, delta }, attrs = new Dictionary<string, object>() { } });
                    return _fast_path_result[0];
                }
                catch (NotOkStatusException ex)
                {
                    throw ex;
                }
                catch (Exception)
                {
                }
                try
                {
                    return adjust_hue_eager_fallback(images, delta, name: name, ctx: _ctx);
                }
                catch (Exception)
                {
                }
            }
            Dictionary<string, object> keywords = new();
            keywords["images"] = images;
            keywords["delta"] = delta;
            var _op = tf.OpDefLib._apply_op_helper("AdjustHue", name, keywords);
            var _result = _op.outputs;
            if (_execute.must_record_gradient())
            {
                object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
                _execute.record_gradient("AdjustHue", _op.inputs, _attrs, _result);
            }
            return _result[0];
        }

        public static Tensor adjust_hue(Tensor images, float delta, string name = null)
            => adjust_hue(images, delta, name: name);

        public static Tensor adjust_hue_eager_fallback(Tensor images, Tensor delta, string name, Context ctx)
        {
            Tensor[] _inputs_flat = new Tensor[] { images, delta};
            object[] _attrs = new object[] { "T", images.dtype };
            var _result = _execute.execute("AdjustHue", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
            if (_execute.must_record_gradient())
            {
                _execute.record_gradient("AdjustHue", _inputs_flat, _attrs, _result);
            }
            return _result[0];
        } 

        public static Tensor adjust_saturation(Tensor images, Tensor scale, string name = null)
        {
            var _ctx = tf.Context;
            if (_ctx.executing_eagerly())
            {
                try
                {
                    var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "AdjustSaturation", name)
                    {
                        args = new object[] { images, scale },
                        attrs = new Dictionary<string, object>() { }
                    });
                    return _fast_path_result[0];
                }
                catch (NotOkStatusException ex)
                {
                    throw ex;
                }
                catch (Exception)
                {
                }
                try
                {
                    return adjust_hue_eager_fallback(images, scale, name: name, ctx: _ctx);
                }
                catch (Exception)
                {
                }
            }
            Dictionary<string, object> keywords = new();
            keywords["images"] = images;
            keywords["scale"] = scale;
            var _op = tf.OpDefLib._apply_op_helper("AdjustSaturation", name, keywords);
            var _result = _op.outputs;
            if (_execute.must_record_gradient())
            {
                object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
                _execute.record_gradient("AdjustSaturation", _op.inputs, _attrs, _result);
            }
            return _result[0];
        }

        public static Tensor adjust_saturation(Tensor images, float scale, string name = null)
            => adjust_saturation(images, ops.convert_to_tensor(scale), name: name);

        public static Tensor adjust_saturation_eager_fallback(Tensor images, Tensor scale, string name, Context ctx)
        {
            Tensor[] _inputs_flat = new Tensor[] { images, scale };
            object[] _attrs = new object[] { "T", images.dtype };
            var _result = _execute.execute("AdjustSaturation", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
            if (_execute.must_record_gradient())
            {
                _execute.record_gradient("AdjustSaturation", _inputs_flat, _attrs, _result);
            }
            return _result[0];
        }

        public static (Tensor, Tensor, Tensor, Tensor) combined_non_max_suppression(Tensor boxes, Tensor scores, Tensor max_output_size_per_class, Tensor max_total_size,
            Tensor iou_threshold, Tensor score_threshold, bool pad_per_class = false, bool clip_boxes = true, string name = null)
        {
            var _ctx = tf.Context;
            if (_ctx.executing_eagerly())
            {
                try
                {
                    var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "CombinedNonMaxSuppression", name){
                            args = new object[] {
                                boxes, scores, max_output_size_per_class, max_total_size, iou_threshold, score_threshold,
                                "pad_per_class", pad_per_class, "clip_boxes", clip_boxes},
                            attrs = new Dictionary<string, object>() { }});
                    return (_fast_path_result[0], _fast_path_result[1], _fast_path_result[2], _fast_path_result[3]);
                }
                catch (NotOkStatusException ex)
                {
                    throw ex;
                }
                catch (Exception)
                {
                }
                try
                {
                    return combined_non_max_suppression_eager_fallback(
                        boxes, scores, max_output_size_per_class, max_total_size, iou_threshold, 
                        score_threshold, pad_per_class, clip_boxes, name, ctx: _ctx);
                }
                catch (Exception)
                {
                }
            }
            Dictionary<string, object> keywords = new();
            keywords["boxes"] = boxes;
            keywords["scores"] = scores;
            keywords["max_output_size_per_class"] = max_output_size_per_class;
            keywords["max_total_size"] = max_total_size;
            keywords["iou_threshold"] = iou_threshold;
            keywords["score_threshold"] = score_threshold;
            keywords["pad_per_class"] = pad_per_class;
            keywords["clip_boxes"] = clip_boxes;

            var _op = tf.OpDefLib._apply_op_helper("CombinedNonMaxSuppression", name, keywords);
            var _result = _op.outputs;
            if (_execute.must_record_gradient())
            {
                object[] _attrs = new object[] { "pad_per_class", _op._get_attr_type("pad_per_class") ,"clip_boxes", _op._get_attr_type("clip_boxes")};
                _execute.record_gradient("CombinedNonMaxSuppression", _op.inputs, _attrs, _result);
            }
            return (_result[0], _result[1], _result[2], _result[3]);
        }

        public static (Tensor, Tensor, Tensor, Tensor) combined_non_max_suppression_eager_fallback(Tensor boxes, Tensor scores, Tensor max_output_size_per_class, Tensor max_total_size,
            Tensor iou_threshold, Tensor score_threshold, bool pad_per_class, bool clip_boxes, string name, Context ctx)
        {
            Tensor[] _inputs_flat = new Tensor[] { boxes, scores, max_output_size_per_class, max_total_size, iou_threshold, score_threshold };
            object[] _attrs = new object[] { "pad_per_class", pad_per_class, "clip_boxes", clip_boxes };
            var _result = _execute.execute("CombinedNonMaxSuppression", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
            if (_execute.must_record_gradient())
            {
                _execute.record_gradient("CombinedNonMaxSuppression", _inputs_flat, _attrs, _result);
            }
            return (_result[0], _result[1], _result[2], _result[3]);
        }

        public static Tensor crop_and_resize(Tensor image, Tensor boxes, Tensor box_ind, Tensor crop_size, string method = "bilinear", float extrapolation_value = 0f, string name = null)
        {
            var _ctx = tf.Context;
            if (_ctx.executing_eagerly())
            {
                try
                {
                    var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "CropAndResize", name) { 
                        args = new object[] { 
                            image, boxes, box_ind, crop_size, "method", method, "extrapolation_value", extrapolation_value }, attrs = new Dictionary<string, object>() { } });
                    return _fast_path_result[0];
                }
                catch (NotOkStatusException ex)
                {
                    throw ex;
                }
                catch (Exception)
                {
                }
                try
                {
                    return crop_and_resize_eager_fallback(
                        image, boxes, box_ind, crop_size, method: method, extrapolation_value: extrapolation_value, name: name, ctx: _ctx);
                }
                catch (Exception)
                {
                }
            }
            Dictionary<string, object> keywords = new();
            keywords["image"] = image;
            keywords["boxes"] = boxes;
            keywords["box_ind"] = box_ind;
            keywords["crop_size"] = crop_size;
            keywords["method"] = method;
            keywords["extrapolation_value"] = extrapolation_value;
            var _op = tf.OpDefLib._apply_op_helper("CropAndResize", name, keywords);
            var _result = _op.outputs;
            if (_execute.must_record_gradient())
            {
                object[] _attrs = new object[] { "T", _op._get_attr_type("T") ,"method", _op._get_attr_type("method") ,
                                                "extrapolation_value", _op.get_attr("extrapolation_value")};
                _execute.record_gradient("CropAndResize", _op.inputs, _attrs, _result);
            }
            return _result[0];
        }

        public static Tensor crop_and_resize_eager_fallback(Tensor image, Tensor boxes, Tensor box_ind, Tensor crop_size, string method, float extrapolation_value, string name, Context ctx)
        {
            if (method is null)
                method = "bilinear";
            //var method_cpmpat = ByteString.CopyFromUtf8(method ?? string.Empty);
            //var extrapolation_value_float = (float)extrapolation_value;

            Tensor[] _inputs_flat = new Tensor[] { image, boxes, box_ind, crop_size, tf.convert_to_tensor(method), tf.convert_to_tensor(extrapolation_value) };
            object[] _attrs = new object[] { "T", image.dtype };
            var _result = _execute.execute("CropAndResize", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
            if (_execute.must_record_gradient())
            {
                _execute.record_gradient("CropAndResize", _inputs_flat, _attrs, _result);
            }
            return _result[0];
        }


        public static Tensor convert_image_dtype(Tensor image, TF_DataType dtype, bool saturate = false, string name = null)
        {
            if (dtype == image.dtype)
                return array_ops.identity(image, name: name);

            return tf_with(ops.name_scope(name, "convert_image", image), scope =>
            {
                name = scope;

                if (image.dtype.is_integer() && dtype.is_integer())
                {
                    throw new NotImplementedException("convert_image_dtype is_integer");
                }
                else if (image.dtype.is_floating() && dtype.is_floating())
                {
                    throw new NotImplementedException("convert_image_dtype is_floating");
                }
                else
                {
                    if (image.dtype.is_integer())
                    {
                        // Converting to float: first cast, then scale. No saturation possible.
                        var cast = math_ops.cast(image, dtype);
                        var scale = 1.0f / image.dtype.max();
                        return math_ops.multiply(cast, scale, name: name);
                    }
                    else
                    {
                        throw new NotImplementedException("convert_image_dtype is_integer");
                    }
                }
            });
        }

        public static Tensor decode_image(Tensor contents,
            long channels = 0,
            TF_DataType dtype = TF_DataType.TF_UINT8,
            bool expand_animations = true,
            string name = null)
                => tf.Context.ExecuteOp("DecodeImage", name,
                    new ExecuteOpArgs(contents).SetAttributes(new
                    {
                        channels,
                        dtype,
                        expand_animations
                    }));

        public static Tensor decode_jpeg(Tensor contents,
            long channels = 0,
            long ratio = 1,
            bool fancy_upscaling = true,
            bool try_recover_truncated = false,
            float acceptable_fraction = 1,
            string dct_method = "",
            string name = null)
                => tf.Context.ExecuteOp("DecodeJpeg", name,
                    new ExecuteOpArgs(contents).SetAttributes(
                    new
                    {
                        channels,
                        ratio,
                        fancy_upscaling,
                        try_recover_truncated,
                        acceptable_fraction,
                        dct_method
                    }));

        public static Tensor decode_gif(Tensor contents,
            string name = null)
        {
            // Add nodes to the TensorFlow graph.
            if (tf.Context.executing_eagerly())
            {
                throw new NotImplementedException("decode_gif");
            }
            else
            {
                var _op = tf.OpDefLib._apply_op_helper("DecodeGif", name: name, args: new
                {
                    contents
                });

                return _op.output;
            }
        }

        public static Tensor decode_png(Tensor contents,
            int channels = 0,
            TF_DataType dtype = TF_DataType.TF_UINT8,
            string name = null)
        {
            // Add nodes to the TensorFlow graph.
            if (tf.Context.executing_eagerly())
            {
                throw new NotImplementedException("decode_png");
            }
            else
            {
                var _op = tf.OpDefLib._apply_op_helper("DecodePng", name: name, args: new
                {
                    contents,
                    channels,
                    dtype
                });

                return _op.output;
            }
        }

        public static Tensor decode_bmp(Tensor contents,
            int channels = 0,
            string name = null)
        {
            // Add nodes to the TensorFlow graph.
            if (tf.Context.executing_eagerly())
            {
                throw new NotImplementedException("decode_bmp");
            }
            else
            {
                var _op = tf.OpDefLib._apply_op_helper("DecodeBmp", name: name, args: new
                {
                    contents,
                    channels
                });

                return _op.output;
            }
        }

        public static Tensor resize_bilinear(Tensor images,
            Tensor size,
            bool align_corners = false,
            bool half_pixel_centers = false,
            string name = null)
                => tf.Context.ExecuteOp("ResizeBilinear", name,
                    new ExecuteOpArgs(images, size).SetAttributes(new
                    {
                        align_corners,
                        half_pixel_centers
                    }));

        public static Tensor resize_bicubic(Tensor images,
            Tensor size,
            bool align_corners = false,
            bool half_pixel_centers = false,
            string name = null)
                => tf.Context.ExecuteOp("ResizeBicubic", name, 
                    new ExecuteOpArgs(images, size).SetAttributes(new { align_corners, half_pixel_centers }));
        
        public static Tensor resize_nearest_neighbor<Tsize>(Tensor images, Tsize size, bool align_corners = false,
            bool half_pixel_centers = false, string name = null)
                => tf.Context.ExecuteOp("ResizeNearestNeighbor", name, 
                    new ExecuteOpArgs(images, size).SetAttributes(new { align_corners, half_pixel_centers }));

        public static Tensor resize_nearest_neighbor_grad(Tensor grads, Tensor size, bool align_corners = false,
            bool half_pixel_centers = false, string name = null)
                => tf.Context.ExecuteOp("ResizeNearestNeighborGrad", name, new ExecuteOpArgs(grads, size)
                {
                    GetGradientAttrs = (op) => new
                    {
                        T = op.get_attr<TF_DataType>("T"),
                        align_corners = op.get_attr<bool>("align_corners"),
                        half_pixel_centers = op.get_attr<bool>("half_pixel_centers")
                    }
                }.SetAttributes(new { align_corners, half_pixel_centers }));
    }
}
