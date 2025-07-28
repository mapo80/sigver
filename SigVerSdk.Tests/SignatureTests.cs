using System;
using System.Collections.Generic;
using System.IO;
using SigVerSdk;
using Xunit;

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
            yield return new object[] { i, folder, gen, forg };
        }
    }

    public static IEnumerable<object[]> GenuinePairs()
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
        var rng = new Random(99);
        for (int i = 0; i < 30; i++)
        {
            var folder = folders[rng.Next(folders.Count)];
            var genuineDir = Path.Combine(baseDir, folder);
            var genuineFiles = Directory.GetFiles(genuineDir);
            if (genuineFiles.Length < 2)
                continue;
            var first = genuineFiles[rng.Next(genuineFiles.Length)];
            var secondIndex = rng.Next(genuineFiles.Length);
            while (secondIndex == Array.IndexOf(genuineFiles, first))
                secondIndex = rng.Next(genuineFiles.Length);
            var second = genuineFiles[secondIndex];
            yield return new object[] { i, folder, first, second };
        }
    }

    [Theory]
    [MemberData(nameof(GenuineForgedPairs))]
    public void GenuineVsForged_ShouldBeDetected(int index, string folderId, string genuinePath, string forgedPath)
    {
        _ = index; // ensure unique test id
        var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../models/signet.onnx"));
        using var verifier = new SigVerifier(modelPath);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var isForgery = verifier.IsForgery(genuinePath, forgedPath, 0.35f);
        sw.Stop();
        Console.WriteLine($"{Path.GetFileName(genuinePath)} vs {Path.GetFileName(forgedPath)} -> detected={isForgery} time={sw.ElapsedMilliseconds}ms");
        Assert.True(isForgery);
    }

    [Theory]
    [MemberData(nameof(GenuinePairs))]
    public void GenuineVsGenuine_ShouldMatch(int index, string folderId, string referencePath, string candidatePath)
    {
        _ = index;
        var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../models/signet.onnx"));
        using var verifier = new SigVerifier(modelPath);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = verifier.Verify(referencePath, candidatePath);
        sw.Stop();
        Console.WriteLine($"{Path.GetFileName(referencePath)} vs {Path.GetFileName(candidatePath)} -> match={result} time={sw.ElapsedMilliseconds}ms");
        Assert.True(result);
    }
}

public static class SigVerifierExtensions
{
public static bool Verify(this SigVerifier verifier, string genuinePath, string candidatePath, float threshold = 0.35f)
    {
        return !verifier.IsForgery(genuinePath, candidatePath, threshold);
    }
}
