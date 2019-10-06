using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Models.ObjectDetection.Core
{
    public class Preprocessor
    {
        public static Tensor[] resize_to_range(ResizeToRangeArgs args)
        {
            var image = args.image;
            var min_dimension = args.min_dimension;
            var max_dimension = args.max_dimension;
            var method = args.method;
            var align_corners = args.align_corners;

            if (image.NDims != 3)
                throw new ValueError("Image should be 3D tensor");

            
            Func<Tensor, Tensor> _resize_landscape_image = (image1) =>
            {
                return tf.image.resize_images(image1,
                    tf.stack(new[] { min_dimension, max_dimension }),
                    method: method,
                        align_corners: align_corners,
                        preserve_aspect_ratio: true);
            };
            Func<Tensor, Tensor> _resize_portrait_image = (image1) =>
            {
                return tf.image.resize_images(image1,
                    tf.stack(new[] { min_dimension, max_dimension }),
                    method: method,
                        align_corners: align_corners,
                        preserve_aspect_ratio: true);
            };

            return tf_with(tf.name_scope("ResizeToRange", values: new { image, min_dimension }), delegate
            {
                Tensor new_image, new_size;

                if (image.TensorShape.is_fully_defined())
                    throw new NotImplementedException("");
                else
                {
                    new_image = tf.cond(
                      tf.less(tf.shape(image)[0], tf.shape(image)[1]),
                      () => _resize_landscape_image(image),
                      () => _resize_portrait_image(image));
                    new_size = tf.shape(new_image);
                }

                if (args.pad_to_max_dimension)
                {
                    throw new NotImplementedException("");
                }

                var result = new List<Tensor> { new_image };
                if (args.masks != null)
                    throw new NotImplementedException("");

                result.Add(new_size);

                return result.ToArray();
            });
        }
    }
}
