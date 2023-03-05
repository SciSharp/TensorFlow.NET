using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.NumPy;

namespace Tensorflow.Keras.UnitTest.Helpers
{
    public class RandomDataSet : DataSetBase
    {
        private Shape _shape;

        public RandomDataSet(Shape shape, int count)
        {
            _shape = shape;
            Debug.Assert(_shape.ndim == 3);
            long[] dims = new long[4];
            dims[0] = count;
            for (int i = 1; i < 4; i++)
            {
                dims[i] = _shape[i - 1];
            }
            Shape s = new Shape(dims);
            Data = np.random.normal(0, 2, s);
            Labels = np.random.uniform(0, 1, (count, 1));
        }
    }
}
