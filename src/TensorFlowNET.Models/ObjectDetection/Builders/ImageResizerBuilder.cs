using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using Tensorflow.Models.ObjectDetection.Core;
using Tensorflow.Models.ObjectDetection.Protos;
using static Tensorflow.Models.ObjectDetection.Protos.ImageResizer;

namespace Tensorflow.Models.ObjectDetection
{
    public class ImageResizerBuilder
    {
        public ImageResizerBuilder()
        {

        }

        /// <summary>
        /// Builds callable for image resizing operations.
        /// </summary>
        /// <param name="image_resizer_config"></param>
        /// <returns></returns>
        public Func<ResizeToRangeArgs, Tensor[]> build(ImageResizer image_resizer_config)
        {
            var image_resizer_oneof = image_resizer_config.ImageResizerOneofCase;
            if (image_resizer_oneof == ImageResizerOneofOneofCase.KeepAspectRatioResizer)
            {
                var keep_aspect_ratio_config = image_resizer_config.KeepAspectRatioResizer;
                if (keep_aspect_ratio_config.MinDimension > keep_aspect_ratio_config.MaxDimension)
                    throw new ValueError("min_dimension > max_dimension");
                var method = _tf_resize_method(keep_aspect_ratio_config.ResizeMethod);
                var per_channel_pad_value = new[] { 0, 0, 0 };
                if (keep_aspect_ratio_config.PerChannelPadValue.Count > 0)
                    throw new NotImplementedException("");
                // per_channel_pad_value = new[] { keep_aspect_ratio_config.PerChannelPadValue. };

                var args = new ResizeToRangeArgs
                {
                    min_dimension = keep_aspect_ratio_config.MinDimension,
                    max_dimension = keep_aspect_ratio_config.MaxDimension,
                    method = method,
                    pad_to_max_dimension = keep_aspect_ratio_config.PadToMaxDimension,
                    per_channel_pad_value = per_channel_pad_value
                };

                Func<ResizeToRangeArgs, Tensor[]> func = (input) =>
                {
                    args.image = input.image;
                    return Preprocessor.resize_to_range(args);
                };

                return func;
            }
            else
            {
                throw new NotImplementedException("");
            }
        }

        private ResizeMethod _tf_resize_method(ResizeType resize_method)
        {
            return (ResizeMethod)(int)resize_method;
        }
    }
}
