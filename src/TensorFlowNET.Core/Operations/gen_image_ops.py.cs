using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class gen_image_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        public Tensor decode_jpeg(Tensor contents,
            int channels = 0,
            int ratio = 1,
            bool fancy_upscaling = true,
            bool try_recover_truncated = false,
            float acceptable_fraction = 1,
            string dct_method = "",
            string name = "")
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

        public Tensor resize_bilinear(Tensor images, int[] size, bool align_corners = false, string name = "")
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
