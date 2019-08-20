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
using static Tensorflow.Python;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class gen_image_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        public Tensor convert_image_dtype(Tensor image, TF_DataType dtype, bool saturate = false, string name= null)
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

        public Tensor decode_jpeg(Tensor contents,
            int channels = 0,
            int ratio = 1,
            bool fancy_upscaling = true,
            bool try_recover_truncated = false,
            float acceptable_fraction = 1,
            string dct_method = "",
            string name = null)
        {
            // Add nodes to the TensorFlow graph.
            if (tf.context.executing_eagerly())
            {
                throw new NotImplementedException("decode_jpeg");
            }
            else
            {
                var _op = _op_def_lib._apply_op_helper("DecodeJpeg", name: name, args: new
                {
                    contents,
                    channels,
                    ratio,
                    fancy_upscaling,
                    try_recover_truncated,
                    acceptable_fraction,
                    dct_method
                });

                return _op.outputs[0];
            }
        }

        public Tensor resize_bilinear(Tensor images, Tensor size, bool align_corners = false, string name = null)
        {
            if (tf.context.executing_eagerly())
            {
                throw new NotImplementedException("resize_bilinear");
            }
            else
            {
                var _op = _op_def_lib._apply_op_helper("ResizeBilinear", name: name, args: new
                {
                    images,
                    size,
                    align_corners
                });

                return _op.outputs[0];
            }
        }
    }
}
