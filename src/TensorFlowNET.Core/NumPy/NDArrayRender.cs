using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace Tensorflow.NumPy
{
    public class NDArrayRender
    {
        public static string ToString(NDArray array, int maxLength = 10)
        {
            Shape shape = array.shape;
            if (shape.IsScalar)
                return Render(array);

            var s = new StringBuilder();
            s.Append("array(");
            Build(s, array, maxLength);
            s.Append(")");
            return s.ToString();
        }

        static void Build(StringBuilder s, NDArray array, int maxLength)
        {
            var shape = array.shape;

            if (shape.Length == 1)
            {
                s.Append("[");
                s.Append(Render(array));
                s.Append("]");
                return;
            }

            var len = shape[0];
            s.Append("[");

            if (len <= maxLength)
            {
                for (int i = 0; i < len; i++)
                {
                    Build(s, array[i], maxLength);
                    if (i < len - 1)
                    {
                        s.Append(", ");
                        s.AppendLine();
                    }
                }
            }
            else
            {
                for (int i = 0; i < maxLength / 2; i++)
                {
                    Build(s, array[i], maxLength);
                    if (i < len - 1)
                    {
                        s.Append(", ");
                        s.AppendLine();
                    }
                }

                s.Append(" ... ");
                s.AppendLine();

                for (int i = (int)len - maxLength / 2; i < len; i++)
                {
                    Build(s, array[i], maxLength);
                    if (i < len - 1)
                    {
                        s.Append(", ");
                        s.AppendLine();
                    }
                }
            }

            s.Append("]");
        }

        static string Render(NDArray array)
        {
            if (array.buffer == IntPtr.Zero)
                return "<null>";

            var dtype = array.dtype;
            var shape = array.shape;

            if (dtype == TF_DataType.TF_STRING)
            {
                if (array.rank == 0)
                    return "'" + string.Join(string.Empty, array.StringBytes()[0]
                        .Take(256)
                        .Select(x => x < 32 || x > 127 ? "\\x" + x.ToString("x") : Convert.ToChar(x).ToString())) + "'";
                else
                    return $"'{string.Join("', '", array.StringData().Take(25))}'";
            }
            else if (dtype == TF_DataType.TF_VARIANT)
            {
                return "<unprintable>";
            }
            else if (dtype == TF_DataType.TF_RESOURCE)
            {
                return "<unprintable>";
            }
            else
            {
                return dtype switch
                {
                    TF_DataType.TF_BOOL => Render(array.ToArray<bool>(), array.shape),
                    TF_DataType.TF_INT8 => Render(array.ToArray<sbyte>(), array.shape),
                    TF_DataType.TF_INT32 => Render(array.ToArray<int>(), array.shape),
                    TF_DataType.TF_INT64 => Render(array.ToArray<long>(), array.shape),
                    TF_DataType.TF_UINT64 => Render(array.ToArray<ulong>(), array.shape),
                    TF_DataType.TF_FLOAT => Render(array.ToArray<float>(), array.shape),
                    TF_DataType.TF_DOUBLE => Render(array.ToArray<double>(), array.shape),
                    _ => Render(array.ToArray<byte>(), array.shape)
                };
            }
        }

        static string Render<T>(T[] array, Shape shape)
        {
            if (array == null)
                return "<null>";

            if (array.Length == 0)
                return "<empty>";

            if (shape.IsScalar)
                return array[0].ToString();

            var display = "";
            if (array.Length <= 10)
                display += string.Join(", ", array);
            else
                display += string.Join(", ", array.Take(5)) + ", ..., " + string.Join(", ", array.Skip(array.Length - 5));
            return display;
        }
    }
}
