using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorFlowNET.Keras.UnitTest;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.UnitTest;

[TestClass]
public class InitializerTest : EagerModeTestBase
{
    [TestMethod]
    public void Orthogonal()
    {
        var initializer = tf.keras.initializers.Orthogonal();
        var values = initializer.Apply(new InitializerArgs((2, 2)));
    }
}
