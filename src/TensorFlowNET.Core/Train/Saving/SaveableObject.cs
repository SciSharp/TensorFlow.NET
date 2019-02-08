using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class SaveableObject
    {
        public Tensor op;
        public SaveSpec[] specs;
        public string name;
        public string device;

        public SaveableObject()
        {

        }

        public SaveableObject(Tensor var, string slice_spec, string name)
        {

        }

        public SaveableObject(Tensor op, SaveSpec[] specs, string name)
        {
            this.op = op;
            this.specs = specs;
            this.name = name;
        }
    }
}
