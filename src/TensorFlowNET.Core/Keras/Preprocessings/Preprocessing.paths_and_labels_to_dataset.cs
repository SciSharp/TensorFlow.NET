using NumSharp;
using System;
using System.Globalization;
using System.Threading.Tasks;
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
            Shape shape = (image_paths.Length, image_size.dims[0], image_size.dims[1], num_channels);
            Console.WriteLine($"Allocating memory for shape{shape}, {NPTypeCode.Float}");
            var data = np.zeros(shape, NPTypeCode.Float);

            for (var i = 0; i < image_paths.Length; i++)
            {
                var image = path_to_image(image_paths[i], image_size, num_channels, interpolation);
                data[i] = image.numpy();
                if (i % 100 == 0)
                    Console.WriteLine($"Filled {i}/{image_paths.Length} data into memory.");
            }

            var img_ds = tf.data.Dataset.from_tensor_slices(data);

            if (label_mode == "int")
            {
                var label_ds = tf.keras.preprocessing.dataset_utils.labels_to_dataset(labels, label_mode, num_classes);
                img_ds = tf.data.Dataset.zip(img_ds, label_ds);
            }
            else
                throw new NotImplementedException("");

            return img_ds;
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
