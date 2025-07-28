using System;
using System.Collections.Generic;
using System.IO;
using SigVerSdk;

namespace SigVerSdk.Tests;

public class SignatureTests
{
    public static IEnumerable<object[]> GenuineForgedPairs()
    {
        var baseDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../data"));
        var users = Directory.GetDirectories(baseDir);
        var folders = new List<string>();
        foreach (var u in users)
        {
            var name = Path.GetFileName(u);
            if (!name.EndsWith("_forg"))
                folders.Add(name);
        }
        var rng = new Random(42);
        for (int i = 0; i < 30; i++)
        {
            var folder = folders[rng.Next(folders.Count)];
            var genuineDir = Path.Combine(baseDir, folder);
            var forgedDir = Path.Combine(baseDir, folder + "_forg");
            var genuineFiles = Directory.GetFiles(genuineDir);
            var forgedFiles = Directory.GetFiles(forgedDir);
            var gen = genuineFiles[rng.Next(genuineFiles.Length)];
            var forg = forgedFiles[rng.Next(forgedFiles.Length)];
            yield return new object[] { folder, gen, forg };
        }
    }

    [Theory]
    [MemberData(nameof(GenuineForgedPairs))]
    public void GenuineVsForged_ShouldBeDetected(string folderId, string genuinePath, string forgedPath)
    {
        var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../models/signet.onnx"));
        using var verifier = new SigVerifier(modelPath);
        var result = verifier.IsForgery(genuinePath, forgedPath);
        Assert.True(result);
    }
}
