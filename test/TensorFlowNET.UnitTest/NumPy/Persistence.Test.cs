using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.NumPy;

namespace TensorFlowNET.UnitTest.NumPy;

/// <summary>
/// https://numpy.org/doc/stable/reference/generated/numpy.save.html
/// </summary>
[TestClass]
public class PersistenceTest : EagerModeTestBase
{
    [TestMethod]
    public void SaveNpy()
    {
        var x = np.arange(10f).reshape((2, 5));
        np.save("arange.npy", x);

        var x2 = np.load("arange.npy");
        Assert.AreEqual(x.shape, x2.shape);
    }

    [TestMethod]
    public void SaveNpz()
    {
        var x = np.arange(10f).reshape((2, 5));
        var y = np.arange(10f).reshape((5, 2));

        np.savez("arange.npz", x, y);
        var z = np.loadz("arange.npz");

        np.savez("arange_named.npz", new { x, y });
        z = np.loadz("arange_named.npz");
        Assert.AreEqual(z["x"].shape, x.shape);
        Assert.AreEqual(z["y"].shape, y.shape);

        np.savez_compressed("arange_compressed.npz", x, y);
        np.savez_compressed("arange_compressed_named.npz", new { x, y });
        z = np.loadz("arange_compressed_named.npz");
        Assert.AreEqual(z["x"].shape, x.shape);
        Assert.AreEqual(z["y"].shape, y.shape);
    }
}
