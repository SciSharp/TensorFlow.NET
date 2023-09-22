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

        public static (int[,], long[]) _remove_long_seq(int maxlen, int[,] seq, long[] label)
        {
            /*Removes sequences that exceed the maximum length.

            Args:
                maxlen: Int, maximum length of the output sequences.
                seq: List of lists, where each sublist is a sequence.
                label: List where each element is an integer.

            Returns:
                    new_seq, new_label: shortened lists for `seq` and `label`.

            */
            var nRow = seq.GetLength(0);
            var nCol = seq.GetLength(1);
            List<int[]> new_seq = new List<int[]>();
            List<long> new_label = new List<long>();

            for (var i = 0; i < nRow; i++)
            {
                if (maxlen < nCol && seq[i, maxlen] != 0)
                    continue;
                int[] sentence = new int[maxlen];
                for (var j = 0; j < maxlen && j < nCol; j++)
                {
                    sentence[j] = seq[i, j];
                }
                new_seq.Add(sentence);
                new_label.Add(label[i]);
            }

            int[,] new_seq_array = new int[new_seq.Count, maxlen];
            long[] new_label_array = new long[new_label.Count];

            for (var i = 0; i < new_seq.Count; i++)
            {
                for (var j = 0; j < maxlen; j++)
                {
                    new_seq_array[i, j] = new_seq[i][j];
                }
            }

            for (var i = 0; i < new_label.Count; i++)
            {
                new_label_array[i] = new_label[i];
            }
            return (new_seq_array, new_label_array);
        }
    }
}
