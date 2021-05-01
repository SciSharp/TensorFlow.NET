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
            if (string.IsNullOrEmpty(cache_dir))
                cache_dir = Path.GetTempPath();
            var datadir_base = cache_dir;
            Directory.CreateDirectory(datadir_base);

            var datadir = Path.Combine(datadir_base, cache_subdir);
            Directory.CreateDirectory(datadir);

            Web.Download(origin, datadir, fname);

            var archive = Path.Combine(datadir, fname);

            if (untar)
                Compress.ExtractTGZ(archive, datadir);
            else if (extract && fname.EndsWith(".gz"))
                Compress.ExtractGZip(archive, datadir);
            else if (extract && fname.EndsWith(".zip"))
                Compress.UnZip(archive, datadir);

            return datadir;
        }
    }
}
