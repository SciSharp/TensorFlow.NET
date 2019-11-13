using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Estimators
{
    public abstract class Exporter<Thyp>
    {
        public abstract void export(Estimator<Thyp> estimator, string export_path, string checkpoint_path);
    }
}
