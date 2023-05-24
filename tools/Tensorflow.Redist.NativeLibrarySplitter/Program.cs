
// =================================================================== //
// This is a tool to split the native .so file of linux gpu library    //
// =================================================================== //

using System.Security.Cryptography;

string filename = "libtensorflow.so";
int count = 5;
SplitFile(filename, count);

static void SplitFile(string filename, int count)
{
    // 打开读取二进制文件的文件流
    using (FileStream input = new FileStream(filename, FileMode.Open, FileAccess.Read))
    {
        long filesize = new FileInfo(filename).Length; // 获取文件大小
        long fragmentSize = (long)(filesize / count + 1); // 计算每个分片的大小

        byte[] buffer = new byte[fragmentSize]; // 设置缓冲区大小
        int bytesRead; // 存储读取长度
        int fragmentIndex = 1; // 分片计数器

        // 使用循环遍历分片并写入相应的文件
        while ((bytesRead = input.Read(buffer, 0, buffer.Length)) > 0)
        {
            string outputFileName = $"{filename}.fragment{fragmentIndex++}";
            using (FileStream output = new FileStream(outputFileName, FileMode.Create, FileAccess.Write))
            {
                output.Write(buffer, 0, bytesRead);
            }
        }

        // 计算整个文件的 SHA-256 哈希值并写入 .sha 文件
        using (SHA256 sha256Hash = SHA256.Create())
        {
            input.Seek(0, SeekOrigin.Begin);
            byte[] hashValue = sha256Hash.ComputeHash(input);

            string shaFileName = $"{filename}.sha";
            using (StreamWriter writer = new StreamWriter(shaFileName, false))
            {
                writer.Write(BitConverter.ToString(hashValue).Replace("-", ""));
            }
        }
    }
}

// Resume the file from fregments. Thanks for the code in TorchSharp!
static void Restitch(string RestitcherPackage)
{
    // !!!!!!!------------------------------NOTE------------------------------------!!!!!!
    // !!!!!!! This code is manually copied into pkg\common\RestitchPackage.targets !!!!!!
    // !!!!!!!------------------------------NOTE------------------------------------!!!!!!
    //
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvv START HERE vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    try
    {
        if (Directory.Exists(RestitcherPackage))
        {
            using (var writer = File.CreateText("obj/tensorflow_redist_build_log.txt"))
            {
                foreach (var p in Directory.EnumerateFiles(RestitcherPackage, "*", SearchOption.AllDirectories))
                {

                    var primaryFile = Path.GetFullPath(p);
                    writer.WriteLine("Found primary file at {0}", primaryFile);

                    // See if there are fragments in the parallel nuget packages. If the primary is 
                    //        some-package-primary\runtimes\....\a.so 
                    //        some-package-primary\runtimes\....\a.so.sha
                    // then the expected fragments are
                    //        some-package-fragment1\fragments\....\a.so 
                    //        some-package-fragment2\fragments\....\a.so 
                    //        some-package-fragment3\fragments\....\a.so 
                    //        some-package-fragment4\fragments\....\a.so 
                    //        some-package-fragment5\fragments\....\a.so 
                    //        some-package-fragment6\fragments\....\a.so 
                    //        some-package-fragment7\fragments\....\a.so 
                    //        some-package-fragment8\fragments\....\a.so 
                    //        some-package-fragment9\fragments\....\a.so 
                    //        some-package-fragment10\fragments\....\a.so 
                    var shaFile = primaryFile + ".sha";
                    var fragmentFile1 = primaryFile.Replace("-primary", "-fragment1").Replace("runtimes", "fragments") + ".fragment1";
                    var fragmentFile2 = primaryFile.Replace("-primary", "-fragment2").Replace("runtimes", "fragments") + ".fragment2";
                    var fragmentFile3 = primaryFile.Replace("-primary", "-fragment3").Replace("runtimes", "fragments") + ".fragment3";
                    var fragmentFile4 = primaryFile.Replace("-primary", "-fragment4").Replace("runtimes", "fragments") + ".fragment4";
                    var fragmentFile5 = primaryFile.Replace("-primary", "-fragment5").Replace("runtimes", "fragments") + ".fragment5";


                    if (File.Exists(fragmentFile1)) writer.WriteLine("Found fragment file at {0}", fragmentFile1);
                    if (File.Exists(fragmentFile2)) writer.WriteLine("Found fragment file at {0}", fragmentFile2);
                    if (File.Exists(fragmentFile3)) writer.WriteLine("Found fragment file at {0}", fragmentFile3);
                    if (File.Exists(fragmentFile4)) writer.WriteLine("Found fragment file at {0}", fragmentFile4);
                    if (File.Exists(fragmentFile5)) writer.WriteLine("Found fragment file at {0}", fragmentFile5);

                    if (File.Exists(fragmentFile1))
                    {
                        var tmpFile = Path.GetTempFileName();

                        {
                            writer.WriteLine("Writing restored primary file at {0}", tmpFile);
                            using (var os = File.OpenWrite(tmpFile))
                            {

                                //writer.WriteLine("Writing bytes from {0} to {1}", primaryFile, tmpFile);
                                //var primaryBytes = File.ReadAllBytes(primaryFile);

                                //os.Write(primaryBytes, 0, primaryBytes.Length);
                                if (File.Exists(fragmentFile1))
                                {
                                    writer.WriteLine("Writing fragment bytes from {0} to {1}", fragmentFile1, tmpFile);
                                    var fragmentBytes1 = File.ReadAllBytes(fragmentFile1);
                                    os.Write(fragmentBytes1, 0, fragmentBytes1.Length);
                                }
                                if (File.Exists(fragmentFile2))
                                {
                                    writer.WriteLine("Writing fragment bytes from {0} to {1}", fragmentFile2, tmpFile);
                                    var fragmentBytes2 = File.ReadAllBytes(fragmentFile2);
                                    os.Write(fragmentBytes2, 0, fragmentBytes2.Length);
                                }
                                if (File.Exists(fragmentFile3))
                                {
                                    writer.WriteLine("Writing fragment bytes from {0} to {1}", fragmentFile3, tmpFile);
                                    var fragmentBytes3 = File.ReadAllBytes(fragmentFile3);
                                    os.Write(fragmentBytes3, 0, fragmentBytes3.Length);
                                }
                                if (File.Exists(fragmentFile4))
                                {
                                    writer.WriteLine("Writing fragment bytes from {0} to {1}", fragmentFile4, tmpFile);
                                    var fragmentBytes4 = File.ReadAllBytes(fragmentFile4);
                                    os.Write(fragmentBytes4, 0, fragmentBytes4.Length);
                                }
                                if (File.Exists(fragmentFile5))
                                {
                                    writer.WriteLine("Writing fragment bytes from {0} to {1}", fragmentFile5, tmpFile);
                                    var fragmentBytes5 = File.ReadAllBytes(fragmentFile5);
                                    os.Write(fragmentBytes5, 0, fragmentBytes5.Length);
                                }
                            }
                        }

                        var shaExpected = File.Exists(shaFile) ? File.ReadAllText(shaFile).ToUpper() : "";
                        writer.WriteLine($"real sha: {shaExpected}");

                        using (var sha256Hash = System.Security.Cryptography.SHA256.Create())
                        {
                            using (var os2 = File.OpenRead(tmpFile))
                            {

                                byte[] bytes = sha256Hash.ComputeHash(os2);
                                var builder = new System.Text.StringBuilder();
                                for (int i = 0; i < bytes.Length; i++)
                                {
                                    builder.Append(bytes[i].ToString("x2"));
                                }
                                var shaReconstituted = builder.ToString().ToUpper();
                                if (shaExpected != shaReconstituted)
                                {
                                    string msg =
                                            $"Error downloading and reviving packages. Reconsituted file contents have incorrect SHA\n\tExpected SHA: ${shaExpected}\n\tActual SHA: ${shaReconstituted}\n\tFile was reconstituted from:"
                                          + $"\n\t{primaryFile} (length ${new FileInfo(primaryFile).Length})"
                                          + (File.Exists(fragmentFile1) ? $"\n\t{fragmentFile1} (length ${new FileInfo(fragmentFile1).Length})" : "")
                                          + (File.Exists(fragmentFile2) ? $"\n\t{fragmentFile2} (length ${new FileInfo(fragmentFile2).Length})" : "")
                                          + (File.Exists(fragmentFile3) ? $"\n\t{fragmentFile3} (length ${new FileInfo(fragmentFile3).Length})" : "")
                                          + (File.Exists(fragmentFile4) ? $"\n\t{fragmentFile4} (length ${new FileInfo(fragmentFile4).Length})" : "")
                                          + (File.Exists(fragmentFile5) ? $"\n\t{fragmentFile5} (length ${new FileInfo(fragmentFile5).Length})" : "");
                                    writer.WriteLine(msg);
                                    throw new Exception(msg);
                                }
                            }

                        }

                        writer.WriteLine("Deleting {0}", primaryFile);
                        File.Delete(primaryFile);
                        if (File.Exists(primaryFile))
                            throw new Exception("wtf?");

                        writer.WriteLine("Moving {0} --> {1}", tmpFile, primaryFile);
                        File.Move(tmpFile, primaryFile);

                        writer.WriteLine("Deleting {0}", fragmentFile1);
                        File.Delete(fragmentFile1);  // free up space and prevent us doing this again 

                        writer.WriteLine("Deleting {0}", fragmentFile2);
                        if (File.Exists(fragmentFile2))
                            File.Delete(fragmentFile2);  // free up space and prevent us doing this again 

                        writer.WriteLine("Deleting {0}", fragmentFile3);
                        if (File.Exists(fragmentFile3))
                            File.Delete(fragmentFile3);  // free up space and prevent us doing this again 

                        writer.WriteLine("Deleting {0}", fragmentFile4);
                        if (File.Exists(fragmentFile4))
                            File.Delete(fragmentFile4);  // free up space and prevent us doing this again 

                        writer.WriteLine("Deleting {0}", fragmentFile5);
                        if (File.Exists(fragmentFile5))
                            File.Delete(fragmentFile5);  // free up space and prevent us doing this again 
                    }
                }
            }
        }
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine(ex.ToString());
        Console.Error.WriteLine(ex.StackTrace);
    }
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ END HERE^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
}