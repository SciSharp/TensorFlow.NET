using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    public partial class NDArray
    {
        public NDArray this[int index]
        {
            get
            {
                return _tensor[index];
            }

            set
            {

            }
        }

        public NDArray this[params int[] index]
        {
            get
            {
                return _tensor[index.Select(x => new Slice(x, x + 1)).ToArray()];
            }

            set
            {

            }
        }

        public NDArray this[params Slice[] slices]
        {
            get
            {
                return _tensor[slices];
            }

            set
            {

            }
        }

        public NDArray this[NDArray mask]
        {
            get
            {
                throw new NotImplementedException("");
            }

            set
            {

            }
        }
    }
}
