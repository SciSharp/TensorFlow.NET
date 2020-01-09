using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Datasets
{
    public class IMDB
    {
        public static ((Tensor, Tensor), (Tensor, Tensor)) load_data(string path= "imdb.npz", int? num_words= null, int skip_top= 0, int? maxlen= null,
                                                            int seed= 113,int start_char= 1, int oov_char= 2, int index_from= 3) => throw new NotImplementedException();

        public static JObject get_word_index(string path= "imdb_word_index.json") => throw new NotImplementedException();
    }
}
