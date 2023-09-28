using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.ManagedAPI
{
    public class RaggedTensorTest :EagerModeTestBase
    {
        [TestMethod]
        public void Test_from_row_lengths()
        {
            var row_lengths = tf.convert_to_tensor(np.array(new int[] { 2, 0, 3, 1, 1 }, TF_DataType.TF_INT64));
            var rp = RowPartition.from_row_lengths(row_lengths, validate: false);
            var rp_row_lengths = rp.row_lengths();
            var rp_nrows = rp.nrows();
            Assert.IsTrue(rp_nrows.ToArray<long>()[0] == rp.nrows().ToArray<long>()[0]);

        }
    }
}
