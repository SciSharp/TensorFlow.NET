using System;
using System.Globalization;
using static Tensorflow.Binding;

namespace Tensorflow.Keras
{
    public partial class Preprocessing
    {
        public Tensor paths_and_labels_to_dataset(string[] image_paths,
            TensorShape image_size,
            int num_channels,
            int[] labels,
            string label_mode,
            int num_classes,
            string interpolation)
        {
            foreach (var image_path in image_paths)
                path_to_image(image_path, image_size, num_channels, interpolation);

            throw new NotImplementedException("");
        }

        Tensor path_to_image(string path, TensorShape image_size, int num_channels, string interpolation)
        {
            var img = tf.io.read_file(path);
            img = tf.image.decode_image(
                img, channels: num_channels, expand_animations: false);
            img = tf.image.resize_images_v2(img, image_size, method: interpolation);
            return img;
        }
    }
}
