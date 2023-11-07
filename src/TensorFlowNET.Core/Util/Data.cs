using OneOf;
using Tensorflow.NumPy;

namespace Tensorflow.Util
{
    /// <summary>
    /// ValidationDataPack is used to pass validation data to fit method.
    /// It can recive data which could be A tuple `(x_val, xy_val)` or `(x_val, y_val, sample_weight_val)` of Numpy arrays.
    /// </summary>
    public class ValidationDataPack
    {
        internal OneOf<NDArray, NDArray[]> val_x;
        internal NDArray val_y;
        internal NDArray val_sample_weight = null;
        public bool val_x_is_array = false;
        public ValidationDataPack((NDArray, NDArray) validation_data)
        {
            this.val_x = validation_data.Item1;
            this.val_y = validation_data.Item2;
        }

        public ValidationDataPack((NDArray, NDArray, NDArray) validation_data)
        {
            this.val_x = validation_data.Item1;
            this.val_y = validation_data.Item2;
            this.val_sample_weight = validation_data.Item3;
        }

        public ValidationDataPack((IEnumerable<NDArray>, NDArray) validation_data)
        {
            this.val_x = validation_data.Item1.ToArray();
            this.val_y = validation_data.Item2;
            val_x_is_array = true;
        }

        public ValidationDataPack((IEnumerable<NDArray>, NDArray, NDArray) validation_data)
        {
            this.val_x = validation_data.Item1.ToArray();
            this.val_y = validation_data.Item2;
            this.val_sample_weight = validation_data.Item3;
            val_x_is_array = true;
        }

        public static implicit operator ValidationDataPack((NDArray, NDArray) validation_data)
            => new ValidationDataPack(validation_data);

        public static implicit operator ValidationDataPack((NDArray, NDArray, NDArray) validation_data)
            => new ValidationDataPack(validation_data);

        public static implicit operator ValidationDataPack((IEnumerable<NDArray>, NDArray) validation_data)
            => new ValidationDataPack(validation_data);

        public static implicit operator ValidationDataPack((IEnumerable<NDArray>, NDArray, NDArray) validation_data)
            => new ValidationDataPack(validation_data);

        public void Deconstruct(out NDArray val_x, out NDArray val_y)
        {
            val_x = this.val_x.AsT0;
            val_y = this.val_y;
        }

        public void Deconstruct(out NDArray val_x, out NDArray val_y, out NDArray val_sample_weight)
        {
            val_x = this.val_x.AsT0;
            val_y = this.val_y;
            val_sample_weight = this.val_sample_weight;
        }

        // add a unuse parameter to make it different from Deconstruct(out NDArray val_x, out NDArray val_y, out NDArray val_sample_weight)
        public void Deconstruct(out NDArray[] val_x_array, out NDArray val_y, out NDArray val_sample_weight, out NDArray unuse)
        {
            val_x_array = this.val_x.AsT1;
            val_y = this.val_y;
            val_sample_weight = this.val_sample_weight;
            unuse = null;
        }
    }
}
