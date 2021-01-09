using System;
using System.IO;

namespace TensorFlowNET.UnitTest
{
    public class TestHelper
    {
        public static string GetFullPathFromDataDir(string fileName)
        {
            var dataDir = GetRootContentDir(Directory.GetCurrentDirectory());
            return Path.Combine(dataDir, fileName);
        }

        static string GetRootContentDir(string dir)
        {
            var path = Path.GetFullPath(Path.Combine(dir, "data"));
            if (Directory.Exists(path))
                return path;
            return GetRootContentDir(Path.GetFullPath(Path.Combine(dir, "..")));
        }
    }
}
