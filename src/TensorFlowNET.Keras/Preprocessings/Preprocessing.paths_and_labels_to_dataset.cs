using System;
using static Tensorflow.Binding;

namespace Tensorflow.Keras
{
    public partial class Preprocessing
    {
        public IDatasetV2 paths_and_labels_to_dataset(string[] image_paths,
            TensorShape image_size,
            int num_channels,
            int[] labels,
            string label_mode,
            int num_classes,
            string interpolation)
        {
            var path_ds = tf.data.Dataset.from_tensor_slices(image_paths);
            var img_ds = path_ds.map(x => path_to_image(x, image_size, num_channels, interpolation));

            if (label_mode == "int")
            {
                var label_ds = dataset_utils.labels_to_dataset(labels, label_mode, num_classes);
                img_ds = tf.data.Dataset.zip(img_ds, label_ds);
            }

            return img_ds;
        }

        Tensor path_to_image(Tensor path, TensorShape image_size, int num_channels, string interpolation)
        {
            var img = tf.io.read_file(path);
            img = tf.image.decode_image(
                img, channels: num_channels, expand_animations: false);
            img = tf.image.resize_images_v2(img, image_size, method: interpolation);
            // img.set_shape((image_size[0], image_size[1], num_channels));
            return img;
        }
    }
}
