using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Estimators
{
    public abstract class Exporter
    {
        public abstract void export(Estimator estimator, string export_path, string checkpoint_path);
    }
}
