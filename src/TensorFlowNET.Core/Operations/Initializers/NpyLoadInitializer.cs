using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.NumPy;

namespace Tensorflow.Operations.Initializers
{
    /// <summary>
    /// An initializer specially used for debugging (to load weights from disk).
    /// </summary>
    class NpyLoadInitializer : IInitializer
    {
        string _path;
        public NpyLoadInitializer(string path) { _path = path; }
        public string ClassName => "";
        public IDictionary<string, object> Config => new Dictionary<string, object>();
        public Tensor Apply(InitializerArgs args)
        {
            return np.load(_path);
        }
    }
}
