using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Tensorflow.UnitTest
{
    public class TestHelper
    {
        public static string GetFullPathFromDataDir(string fileName)
        {
            var dir = Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..", "..", "..", "data");
            return Path.GetFullPath(Path.Combine(dir, fileName));
        } 
    }
}
