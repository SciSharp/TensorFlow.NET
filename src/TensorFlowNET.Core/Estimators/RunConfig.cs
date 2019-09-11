using System;

namespace Tensorflow.Estimators
{
    public class RunConfig
    {
        public string model_dir { get; set; }
        public ConfigProto session_config { get; set; }

        public RunConfig(string model_dir)
        {
            this.model_dir = model_dir;
        }
    }
}
