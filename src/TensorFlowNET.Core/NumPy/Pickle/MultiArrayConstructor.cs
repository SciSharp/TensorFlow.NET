using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Text;
using Razorvine.Pickle;
using Razorvine.Pickle.Objects;

namespace Tensorflow.NumPy.Pickle
{
    /// <summary>
    /// Creates multiarrays of objects. Returns a primitive type multiarray such as int[][] if 
    /// the objects are ints, etc. 
    /// </summary>
    [SuppressMessage("ReSharper", "InconsistentNaming")]
    [SuppressMessage("ReSharper", "MemberCanBePrivate.Global")]
    [SuppressMessage("ReSharper", "MemberCanBeMadeStatic.Global")]
    public class MultiArrayConstructor : IObjectConstructor
    {
        public object construct(object[] args)
        {
            if (args.Length != 3)
                throw new InvalidArgumentError($"Invalid number of arguments in MultiArrayConstructor._reconstruct. Expected three arguments. Given {args.Length} arguments.");

            var types = (ClassDictConstructor)args[0];
            if (types.module != "numpy" || types.name != "ndarray")
                throw new RuntimeError("_reconstruct: First argument must be a sub-type of ndarray");

            var arg1 = (object[])args[1];
            var dims = new int[arg1.Length];
            for (var i = 0; i < arg1.Length; i++)
            {
                dims[i] = (int)arg1[i];
            }
            var shape = new Shape(dims);

            TF_DataType dtype;
            string identifier;
            if (args[2].GetType() == typeof(string))
                identifier = (string)args[2];
            else
                identifier = Encoding.UTF8.GetString((byte[])args[2]);
            switch (identifier)
            {
                case "u": dtype = np.uint32; break;
                case "c": dtype = np.complex_; break;
                case "f": dtype = np.float32; break;
                case "b": dtype = np.@bool; break;
                default: throw new NotImplementedException($"Unsupported data type: {args[2]}");
            }
            return new MultiArrayPickleWarpper(shape, dtype);
        }
    }
}
