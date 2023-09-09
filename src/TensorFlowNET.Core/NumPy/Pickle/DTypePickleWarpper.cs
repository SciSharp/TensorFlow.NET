using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.NumPy.Pickle
{
    public class DTypePickleWarpper
    {
        TF_DataType dtype { get; set; }
        public DTypePickleWarpper(TF_DataType dtype)
        {
            this.dtype = dtype;
        }
        public void __setstate__(object[] args) { }
        public static implicit operator TF_DataType(DTypePickleWarpper dTypeWarpper)
        {
            return dTypeWarpper.dtype;
        }
    }
}
