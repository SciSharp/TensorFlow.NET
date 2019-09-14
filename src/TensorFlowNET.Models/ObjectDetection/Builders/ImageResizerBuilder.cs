using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Models.ObjectDetection.Protos;
using static Tensorflow.Models.ObjectDetection.Protos.ImageResizer;

namespace Tensorflow.Models.ObjectDetection
{
    public class ImageResizerBuilder
    {
        public ImageResizerBuilder()
        {

        }

        public Action build(ImageResizer image_resizer_config)
        {
            var image_resizer_oneof = image_resizer_config.ImageResizerOneofCase;
            if (image_resizer_oneof == ImageResizerOneofOneofCase.KeepAspectRatioResizer)
            {
                var keep_aspect_ratio_config = image_resizer_config.KeepAspectRatioResizer;
                var method = _tf_resize_method(keep_aspect_ratio_config.ResizeMethod);
                var per_channel_pad_value = new[] { 0, 0, 0 };
                //if (keep_aspect_ratio_config.PerChannelPadValue != null)
                    //per_channel_pad_value = new[] { keep_aspect_ratio_config.PerChannelPadValue };
            }
            else
            {
                throw new NotImplementedException("");
            }

            return null;
        }

        private ResizeType _tf_resize_method(ResizeType resize_method)
        {
            return resize_method;
        }
    }
}
