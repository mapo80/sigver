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
        var images = new[] { "a1.png", "a2.png", "b1.png", "b2.png" };
        var times = new List<long>();
        foreach (var img in images)
        {
            var path = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, $"../../../../data/{img}"));
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
        var pairs = new[]
        {
            ("a1.png", "a2.png", false),
            ("a1.png", "b1.png", true),
            ("a1.png", "b2.png", true)
        };

        var times = new List<long>();
        foreach (var pair in pairs)
        {
            var refPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, $"../../../../data/{pair.Item1}"));
            var candPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, $"../../../../data/{pair.Item2}"));
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

