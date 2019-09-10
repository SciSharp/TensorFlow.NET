using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Estimators
{
    public class Estimator : IObjectLife
    {
        public RunConfig config;

        public Estimator(RunConfig config)
        {
            this.config = config;
        }

        public void __init__()
        {
            throw new NotImplementedException();
        }

        public void __enter__()
        {
            throw new NotImplementedException();
        }

        public void __del__()
        {
            throw new NotImplementedException();
        }

        public void __exit__()
        {
            throw new NotImplementedException();
        }

        public void Dispose()
        {
            throw new NotImplementedException();
        }
    }
}
