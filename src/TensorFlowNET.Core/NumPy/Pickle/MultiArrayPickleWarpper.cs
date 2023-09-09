using Newtonsoft.Json.Linq;
using Serilog.Debugging;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.NumPy.Pickle
{
    public class MultiArrayPickleWarpper
    {
        public Shape reconstructedShape { get; set; }
        public TF_DataType reconstructedDType { get; set; }
        public NDArray reconstructedNDArray { get; set; }
        public Array reconstructedMultiArray { get; set; }
        public MultiArrayPickleWarpper(Shape shape, TF_DataType dtype)
        {
            reconstructedShape = shape;
            reconstructedDType = dtype;
        }
        public void __setstate__(object[] args)
        {
            if (args.Length != 5)
                throw new InvalidArgumentError($"Invalid number of arguments in NDArray.__setstate__. Expected five arguments. Given {args.Length} arguments.");

            var version = (int)args[0]; // version

            var arg1 = (object[])args[1];
            var dims = new int[arg1.Length];
            for (var i = 0; i < arg1.Length; i++)
            {
                dims[i] = (int)arg1[i];
            }
            var _ShapeLike = new Shape(dims); // shape

            TF_DataType _DType_co = (DTypePickleWarpper)args[2]; // DType

            var F_continuous = (bool)args[3]; // F-continuous
            if (F_continuous)
                throw new InvalidArgumentError("Fortran Continuous memory layout is not supported. Please use C-continuous layout or check the data format.");

            var data = args[4]; // Data
            /*
             * If we ever need another pickle format, increment the version
             * number. But we should still be able to handle the old versions.
             */
            if (version < 0 || version > 4)
                throw new ValueError($"can't handle version {version} of numpy.dtype pickle");

            // TODO: Implement the missing details and checks from the official Numpy C code here.
            // https://github.com/numpy/numpy/blob/2f0bd6e86a77e4401d0384d9a75edf9470c5deb6/numpy/core/src/multiarray/descriptor.c#L2761

            if (data.GetType() == typeof(ArrayList))
            {
                Reconstruct((ArrayList)data);
            }
            else
                throw new NotImplementedException("");
        }
        private void Reconstruct(ArrayList arrayList)
        {
            int ndim = 1;
            var subArrayList = arrayList;
            while (subArrayList.Count > 0 && subArrayList[0] != null && subArrayList[0].GetType() == typeof(ArrayList))
            {
                subArrayList = (ArrayList)subArrayList[0];
                ndim += 1;
            }
            var type = subArrayList[0].GetType();
            if (type == typeof(int))
            {
                if (ndim == 1)
                {
                    int[] list = (int[])arrayList.ToArray(typeof(int));
                    Shape shape = new Shape(new int[] { arrayList.Count });
                    reconstructedMultiArray = list;
                    reconstructedNDArray = new NDArray(list, shape);
                }
                if (ndim == 2)
                {
                    int secondDim = 0;
                    foreach (ArrayList subArray in arrayList)
                    {
                        secondDim = subArray.Count > secondDim ? subArray.Count : secondDim;
                    }
                    int[,] list = new int[arrayList.Count, secondDim];
                    for (int i = 0; i < arrayList.Count; i++)
                    {
                        var subArray = (ArrayList?)arrayList[i];
                        if (subArray == null)
                            throw new NullReferenceException("");
                        for (int j = 0; j < subArray.Count; j++)
                        {
                            var element = subArray[j];
                            if (element == null)
                                throw new NoNullAllowedException("the element of ArrayList cannot be null.");
                            list[i, j] = (int)element;
                        }
                    }
                    Shape shape = new Shape(new int[] { arrayList.Count, secondDim });
                    reconstructedMultiArray = list;
                    reconstructedNDArray = new NDArray(list, shape);
                }
                if (ndim > 2)
                    throw new NotImplementedException("can't handle ArrayList with more than two dimensions.");
            }
            else
                throw new NotImplementedException("");
        }
        public static implicit operator Array(MultiArrayPickleWarpper arrayWarpper)
        {
            return arrayWarpper.reconstructedMultiArray;
        }
        public static implicit operator NDArray(MultiArrayPickleWarpper arrayWarpper)
        {
            return arrayWarpper.reconstructedNDArray;
        }
    }
}
