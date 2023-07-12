using static Tensorflow.KerasApi;
using static Tensorflow.Binding;
using Tensorflow.NumPy; 

namespace Tensorflow.Keras
{
    public partial class Preprocessing
    {
        public static string[] WHITELIST_FORMATS = new[] { ".bmp", ".gif", ".jpeg", ".jpg", ".png" };

        /// <summary>
        /// Function that calculates the classification statistics for a given array of classified data. 
        /// The function takes an array of classified data as input and returns a dictionary containing the count and percentage of each class in the input array. 
        /// This function can be used to analyze the distribution of classes in a dataset or to evaluate the performance of a classification model.
        /// </summary>
        /// <remarks>
        /// code from copilot
        /// </remarks>
        /// <param name="label_ids"></param>
        /// <param name="label_class_names"></param>
        Dictionary<string, double> get_classification_statistics(int[] label_ids, string[] label_class_names)
        {
            var countDict = label_ids.GroupBy(x => x)
                        .ToDictionary(g => g.Key, g => g.Count());
            var totalCount = label_ids.Length;
            var ratioDict = label_class_names.ToDictionary(name => name,
                                                    name =>
                                                    (double)(countDict.ContainsKey(Array.IndexOf(label_class_names, name)) 
                                                    ? countDict[Array.IndexOf(label_class_names, name)] : 0)
                                                    / totalCount);

            print("Classification statistics:");
            foreach (string labelName in label_class_names)
            {
                double ratio = ratioDict[labelName];
                print($"{labelName}: {ratio * 100:F2}%");
            }

            return ratioDict;
        }

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
            Shape image_size = null,
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
                labels,
                formats: WHITELIST_FORMATS,
                class_names: class_names,
                shuffle: shuffle,
                seed: seed,
                follow_links: follow_links);

            (image_paths, label_list) = keras.preprocessing.dataset_utils.get_training_or_validation_split(image_paths, label_list, validation_split, subset);
            get_classification_statistics(label_list, class_name_list);

            var dataset = paths_and_labels_to_dataset(image_paths, image_size, num_channels, label_list, label_mode, class_name_list.Length, interpolation);
            if (shuffle)
                dataset = dataset.shuffle(batch_size * 8, seed: seed);
            dataset = dataset.batch(batch_size);
            dataset.class_names = class_name_list;
            return dataset;
        }

        public IDatasetV2 text_dataset_from_directory(string directory,
            string labels = "inferred",
            string label_mode = "int",
            string[] class_names = null,
            int batch_size = 32,
            bool shuffle = true,
            int max_length = -1,
            int? seed = null,
            float validation_split = 0.2f,
            string subset = null,
            bool follow_links = false)
        {
            var (file_paths, label_list, class_name_list) = dataset_utils.index_directory(
               directory,
               labels,
               formats: new[] { ".txt" },
               class_names: class_names,
               shuffle: shuffle,
               seed: seed,
               follow_links: follow_links);

            (file_paths, label_list) = dataset_utils.get_training_or_validation_split(
                file_paths, label_list, validation_split, subset);

            var dataset = paths_and_labels_to_dataset(file_paths, label_list, label_mode, class_name_list.Length);
            if (shuffle)
                dataset = dataset.shuffle(batch_size * 8, seed: seed);
            dataset = dataset.batch(batch_size);
            dataset.class_names = class_name_list;
            return dataset;
        }

        /// <summary>
        /// Creates a dataset of sliding windows over a timeseries provided as array.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="sequence_length"></param>
        /// <param name="sequence_stride"></param>
        /// <param name="sampling_rate"></param>
        /// <param name="batch_size"></param>
        /// <param name="shuffle"></param>
        /// <returns></returns>
        public IDatasetV2 timeseries_dataset_from_array(Tensor data, int sequence_length,
            int sequence_stride = 1, int sampling_rate = 1, int batch_size = 128,
            bool shuffle = false, int seed = (int)1e6, int start_index = 0, int? end_index = null)
        {
            if (!end_index.HasValue)
                end_index = len(data);

            var num_seqs = end_index.Value - start_index - (sequence_length * sampling_rate) + 1;
            var index_dtype = num_seqs < 2147483647 ? tf.int32 : tf.int64;
            var start_positions = np.arange(0, num_seqs, sequence_stride);
            if (shuffle)
            {
                tf.set_random_seed(seed);
                np.random.shuffle(start_positions);
            }

            var sequence_length_tensor = constant_op.constant(sequence_length, dtype: index_dtype);
            var sampling_rate_tensor = constant_op.constant(sampling_rate, dtype: index_dtype);

            var start_positions_tensor = tf.constant(start_positions);
            var positions_ds = tf.data.Dataset.from_tensors(start_positions_tensor).repeat();
            var r = tf.data.Dataset.range(len(start_positions));
            var z = tf.data.Dataset.zip(r, positions_ds);
            var indices = z.map(m =>
            {
                var (i, positions) = m;
                return tf.range(positions.Single[i], positions.Single[i] + sequence_length_tensor * sampling_rate_tensor, sampling_rate_tensor);
            }, num_parallel_calls: -1);
            var dataset = sequences_from_indices(data, indices, start_index, end_index);

            if (shuffle)
                dataset = dataset.shuffle(buffer_size: batch_size * 8, seed: seed);
            dataset = dataset.batch(batch_size);
            return dataset;
        }

        IDatasetV2 sequences_from_indices(Tensor array, IDatasetV2 indices_ds, int start_index, int? end_index)
        {
            var dataset = tf.data.Dataset.from_tensors(array[new Slice(start: start_index, stop: end_index)]);
            dataset = tf.data.Dataset.zip(dataset.repeat(), indices_ds)
                .map(x =>
                {
                    var (steps, indx) = x;
                    return array_ops.gather(steps, indx);
                }, num_parallel_calls: -1);
            return dataset;
        }
    }
}
