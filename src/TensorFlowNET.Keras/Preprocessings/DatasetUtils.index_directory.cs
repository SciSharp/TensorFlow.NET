using NumSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Tensorflow.Keras.Preprocessings
{
    public partial class DatasetUtils
    {
        /// <summary>
        /// Make list of all files in the subdirs of `directory`, with their labels.
        /// </summary>
        /// <param name="directory"></param>
        /// <param name="labels"></param>
        /// <param name="formats"></param>
        /// <param name="class_names"></param>
        /// <param name="shuffle"></param>
        /// <param name="seed"></param>
        /// <param name="follow_links"></param>
        /// <returns>
        /// file_paths, labels, class_names
        /// </returns>
        public (string[], int[], string[]) index_directory(string directory,
            string labels,
            string[] formats = null,
            string[] class_names = null,
            bool shuffle = true,
            int? seed = null,
            bool follow_links = false)
        {
            var label_list = new List<int>();
            var file_paths = new List<string>();

            var class_dirs = Directory.GetDirectories(directory);
            class_names = class_dirs.Select(x => x.Split(Path.DirectorySeparatorChar).Last()).ToArray();

            for (var label = 0; label < class_dirs.Length; label++)
            {
                var files = Directory.GetFiles(class_dirs[label]);
                file_paths.AddRange(files);
                label_list.AddRange(Enumerable.Range(0, files.Length).Select(x => label));
            }

            var return_labels = label_list.Select(x => x).ToArray();
            var return_file_paths = file_paths.Select(x => x).ToArray();

            if (shuffle)
            {
                if (!seed.HasValue)
                    seed = np.random.randint((long)1e6);
                var random_index = np.arange(label_list.Count);
                var rng = np.random.RandomState(seed.Value);
                rng.shuffle(random_index);
                var index = random_index.ToArray<int>();

                for (int i = 0; i < label_list.Count; i++)
                {
                    return_labels[i] = label_list[index[i]];
                    return_file_paths[i] = file_paths[index[i]];
                }
            }

            Binding.tf_output_redirect.WriteLine($"Found {return_file_paths.Length} files belonging to {class_names.Length} classes.");
            return (return_file_paths, return_labels, class_names);
        }
    }
}
