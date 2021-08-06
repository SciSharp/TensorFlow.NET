using System;
using System.IO;
using static Tensorflow.Binding;
using Tensorflow.NumPy;

namespace Tensorflow.Keras
{
    public partial class Preprocessing
    {
        public IDatasetV2 paths_and_labels_to_dataset(string[] image_paths,
            Shape image_size,
            int num_channels,
            int[] labels,
            string label_mode,
            int num_classes,
            string interpolation)
        {
            // option 1: will load all images into memory, not efficient
            var images = np.zeros((image_paths.Length, image_size[0], image_size[1], num_channels), np.float32);
            for (int i = 0; i < len(images); i++)
            {
                var img = tf.io.read_file(image_paths[i]);
                img = tf.image.decode_image(
                    img, channels: num_channels, expand_animations: false);
                var resized_image = tf.image.resize_images_v2(img, image_size, method: interpolation);
                images[i] = resized_image.numpy();
                tf_output_redirect.WriteLine(image_paths[i]);
            };

            // option 2: dynamic load, but has error, need to fix
            /* var path_ds = tf.data.Dataset.from_tensor_slices(image_paths);
            var img_ds = path_ds.map(x => path_to_image(x, image_size, num_channels, interpolation));*/

            var img_ds = tf.data.Dataset.from_tensor_slices(images);
            if (label_mode == "int")
            {
                var label_ds = dataset_utils.labels_to_dataset(labels, label_mode, num_classes);
                img_ds = tf.data.Dataset.zip(img_ds, label_ds);
            }

            return img_ds;
        }

        Tensor path_to_image(Tensor path, Shape image_size, int num_channels, string interpolation)
        {
            var img = tf.io.read_file(path);
            img = tf.image.decode_image(
                img, channels: num_channels, expand_animations: false);
            img = tf.image.resize_images_v2(img, image_size, method: interpolation);
            // img.set_shape((image_size[0], image_size[1], num_channels));
            return img;
        }

        public IDatasetV2 paths_and_labels_to_dataset(string[] image_paths,
            int[] labels,
            string label_mode,
            int num_classes,
            int max_length = -1)
        {
            var path_ds = tf.data.Dataset.from_tensor_slices(image_paths);
            var string_ds = path_ds.map(x => path_to_string_content(x, max_length));

            if (label_mode == "int")
            {
                var label_ds = dataset_utils.labels_to_dataset(labels, label_mode, num_classes);
                string_ds = tf.data.Dataset.zip(string_ds, label_ds);
            }

            return string_ds;
        }

        Tensor path_to_string_content(Tensor path, int max_length)
        {
            var txt = tf.io.read_file(path);
            if (max_length > -1)
                txt = tf.strings.substr(txt, 0, max_length);
            return txt;
        }
    }
}
