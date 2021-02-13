using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Utils
{
    public class KerasUtils
    {
        /// <summary>
        /// Downloads a file from a URL if it not already in the cache.
        /// </summary>
        /// <param name="fname">Name of the file</param>
        /// <param name="origin">Original URL of the file</param>
        /// <param name="untar"></param>
        /// <param name="md5_hash"></param>
        /// <param name="file_hash"></param>
        /// <param name="cache_subdir"></param>
        /// <param name="hash_algorithm"></param>
        /// <param name="extract"></param>
        /// <param name="archive_format"></param>
        /// <param name="cache_dir"></param>
        /// <returns></returns>
        public string get_file(string fname, string origin,
            bool untar = false,
            string md5_hash = null,
            string file_hash = null,
            string cache_subdir = "datasets",
            string hash_algorithm = "auto",
            bool extract = false,
            string archive_format = "auto",
            string cache_dir = null)
            => data_utils.get_file(fname, origin, 
                untar: untar,
                md5_hash: md5_hash,
                file_hash: file_hash,
                cache_subdir: cache_subdir,
                hash_algorithm: hash_algorithm,
                extract: extract,
                archive_format: archive_format,
                cache_dir: cache_dir);
    }
}
