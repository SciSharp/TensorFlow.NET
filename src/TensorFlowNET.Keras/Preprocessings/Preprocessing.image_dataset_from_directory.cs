using static Tensorflow.KerasApi;

namespace Tensorflow.Keras
{
    public partial class Preprocessing
    {
        public static string[] WHITELIST_FORMATS = new[] { ".bmp", ".gif", ".jpeg", ".jpg", ".png" };

        /// <summary>
        /// Generates a `tf.data.Dataset` from image files in a directory.
        /// https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory
        /// </summary>
        /// <param name="directory">Directory where the data is located.</param>
        /// <param name="labels"></param>
        /// <param name="label_mode"></param>
        /// <param name="class_names"></param>
        /// <param name="color_mode"></param>
        /// <param name="batch_size"></param>
        /// <param name="image_size"></param>
        /// <param name="shuffle"></param>
        /// <param name="seed"></param>
        /// <param name="validation_split"></param>
        /// <param name="subset"></param>
        /// <param name="interpolation"></param>
        /// <param name="follow_links"></param>
        /// <returns></returns>
        public IDatasetV2 image_dataset_from_directory(string directory,
            string labels = "inferred",
            string label_mode = "int",
            string[] class_names = null,
            string color_mode = "rgb",
            int batch_size = 32,
            TensorShape image_size = null,
            bool shuffle = true,
            int? seed = null,
            float validation_split = 0.2f,
            string subset = null,
            string interpolation = "bilinear",
            bool follow_links = false)
        {
            int num_channels = 0;
            if (color_mode == "rgb")
                num_channels = 3;
            
            var (image_paths, label_list, class_name_list) = keras.preprocessing.dataset_utils.index_directory(directory,
                formats: WHITELIST_FORMATS,
                class_names: class_names,
                shuffle: shuffle,
                seed: seed,
                follow_links: follow_links);

            (image_paths, label_list) = keras.preprocessing.dataset_utils.get_training_or_validation_split(image_paths, label_list, validation_split, subset);

            var dataset = paths_and_labels_to_dataset(image_paths, image_size, num_channels, label_list, label_mode, class_name_list.Length, interpolation);
            if (shuffle)
                dataset = dataset.shuffle(batch_size * 8, seed: seed);
            dataset = dataset.batch(batch_size);
            return dataset;
        }
    }
}
