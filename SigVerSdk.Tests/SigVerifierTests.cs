using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using SigVerSdk;
using System.Diagnostics;

namespace SigVerSdk.Tests;

public class SigVerifierTests
{
    [Fact]
    public void ExtractFeaturesRunsOnSampleImages()
    {
        var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../models/signet.onnx"));
        using var verifier = new SigVerifier(modelPath);
        var images = new[]
        {
            Path.Combine("001", "001_01.PNG"),
            Path.Combine("001", "001_02.PNG"),
            Path.Combine("002", "002_01.PNG"),
            Path.Combine("002", "002_02.PNG")
        };
        var times = new List<long>();
        var dataDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../data"));
        foreach (var img in images)
        {
            var path = Path.Combine(dataDir, img);
            var sw = Stopwatch.StartNew();
            var output = verifier.ExtractFeatures(path);
            sw.Stop();
            Assert.Equal(2048, output.Length);
            times.Add(sw.ElapsedMilliseconds);
        }
        var avg = times.Average();
        Console.WriteLine($"Average inference time: {avg} ms");
    }

    [Fact]
    public void DetectForgeryUsingDistance()
    {
        var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../models/signet.onnx"));
        using var verifier = new SigVerifier(modelPath);
        var dataDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../data"));
        var pairs = new[]
        {
            (Path.Combine("001", "001_01.PNG"), Path.Combine("001", "001_02.PNG"), false),
            (Path.Combine("001", "001_01.PNG"), Path.Combine("002", "002_01.PNG"), true),
            (Path.Combine("001", "001_01.PNG"), Path.Combine("002", "002_02.PNG"), true)
        };

        var times = new List<long>();
        foreach (var pair in pairs)
        {
            var refPath = Path.Combine(dataDir, pair.Item1);
            var candPath = Path.Combine(dataDir, pair.Item2);
            var sw = Stopwatch.StartNew();
            var isForged = verifier.IsForgery(refPath, candPath);
            sw.Stop();
            Assert.Equal(pair.Item3, isForged);
            times.Add(sw.ElapsedMilliseconds);
        }
        var avg = times.Average();
        Console.WriteLine($"Average verification time: {avg} ms");
    }
}

