using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Datasets
{
    public class Reuters
    {
        public static ((Tensor, Tensor), (Tensor, Tensor)) load_data(string path = "reuters.npz", int? num_words= null, int skip_top= 0,
                                                int? maxlen= null,float test_split= 0.2f, int seed= 113,int start_char= 1,int oov_char= 2,int index_from= 3) => throw new NotImplementedException();
    }
}
