using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class random_ops
    {
        public static Tensor random_normal(int[] shape, 
            float mean = 0.0f, 
            float stddev = 1.0f, 
            TF_DataType dtype = TF_DataType.TF_FLOAT, 
            int? seed = null, 
            string name = null)
        {
            return Python.with<ops.name_scope, Tensor>(new ops.name_scope(name, "random_normal", new object[] { shape, mean, stddev }), scope =>
            {
                var shape_tensor = _ShapeTensor(shape);
                var mean_tensor = ops.convert_to_tensor(mean, dtype: dtype, name: "mean");
                var stddev_tensor = ops.convert_to_tensor(stddev, dtype: dtype, name = "stddev");
                var (seed1, seed2) = random_seed.get_seed(seed);
                var rnd = gen_random_ops.random_standard_normal(shape_tensor, dtype: dtype, seed: seed1, seed2: seed2);
                var mul = rnd * stddev_tensor;
                var value = math_ops.add(mul, mean_tensor, name: name);
                return value;
            });
        }

        private static Tensor _ShapeTensor(int[] shape)
        {
            return ops.convert_to_tensor(shape, name: "shape");
        }
    }
}

