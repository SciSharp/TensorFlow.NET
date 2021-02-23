using System;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Tensorflow.Keras.Utils
{
    public class data_utils
    {
        public static string get_file(string fname, string origin,
            bool untar = false,
            string md5_hash = null,
            string file_hash = null,
            string cache_subdir = "datasets",
            string hash_algorithm = "auto",
            bool extract = false,
            string archive_format = "auto",
            string cache_dir = null)
        {
            var datadir_base = cache_dir;
            Directory.CreateDirectory(datadir_base);

            var datadir = Path.Combine(datadir_base, cache_subdir);
            Directory.CreateDirectory(datadir);

            Web.Download(origin, datadir, fname);

            if (untar)
                Compress.ExtractTGZ(Path.Combine(datadir_base, fname), datadir_base);
            else if (extract)
                Compress.ExtractGZip(Path.Combine(datadir_base, fname), datadir_base);

            return datadir;
        }
    }
}
