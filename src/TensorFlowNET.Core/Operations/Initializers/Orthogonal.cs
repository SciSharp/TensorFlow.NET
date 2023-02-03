using System;
using System.Collections.Generic;

namespace Tensorflow.Operations.Initializers
{
    public class Orthogonal : IInitializer
    {
        private readonly Dictionary<string, object> _config;

        public string ClassName => "Orthogonal";
        public IDictionary<string, object> Config => throw new NotImplementedException();
        public Tensor Apply(InitializerArgs args)
        {
            throw new NotImplementedException();
        }
    }
}
