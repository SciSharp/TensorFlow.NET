using Microsoft.VisualStudio.TestTools.UnitTesting;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.UnitTest;

[TestClass]
public class InitializerTest : EagerModeTestBase
{
    [TestMethod]
    public void Orthogonal()
    {
        var initializer = tf.keras.initializers.Orthogonal();
        var values = initializer.Apply(new Tensorflow.InitializerArgs((2, 2)));
    }
}
