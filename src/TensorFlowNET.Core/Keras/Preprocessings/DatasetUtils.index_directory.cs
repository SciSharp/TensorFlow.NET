using System;
using System.Collections.Generic;
using System.Text;

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
            string[] formats,
            string class_names = null,
            bool shuffle = true,
            int? seed = null,
            bool follow_links = false)
        {
            throw new NotImplementedException("");
        }
    }
}
