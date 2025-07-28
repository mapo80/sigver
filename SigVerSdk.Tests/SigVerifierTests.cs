using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using SigVerSdk;
using System.Diagnostics;
using OpenCvSharp;

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

    [Fact]
    public void PreprocessingMatchesPython()
    {
        var root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../"));
        var dataImg = Path.Combine(root, "data", "001", "001_01.PNG");
        var pythonScript = Path.Combine(root, "scripts", "save_preprocessed.py");
        var tmpDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tmpDir);

        var psi = new ProcessStartInfo("python", $"{pythonScript} {tmpDir} {dataImg}")
        {
            WorkingDirectory = root,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        psi.Environment["PYTHONPATH"] = root;
        using (var proc = Process.Start(psi)!)
        {
            proc.WaitForExit();
            Assert.Equal(0, proc.ExitCode);
        }

        var pyFile = Path.Combine(tmpDir, "001_01_py.png");
        var dotnetFile = Path.Combine(tmpDir, "001_01_dotnet.png");

        var modelPath = Path.Combine(root, "models", "signet.onnx");
        using var verifier = new SigVerifier(modelPath);
        verifier.SavePreprocessed(dataImg, dotnetFile);

        using var pyMat = Cv2.ImRead(pyFile, ImreadModes.Grayscale);
        using var netMat = Cv2.ImRead(dotnetFile, ImreadModes.Grayscale);

        Assert.Equal(pyMat.Width, netMat.Width);
        Assert.Equal(pyMat.Height, netMat.Height);
        for (int y = 0; y < pyMat.Rows; y++)
        {
            for (int x = 0; x < pyMat.Cols; x++)
            {
                Assert.Equal(pyMat.At<byte>(y, x), netMat.At<byte>(y, x));
            }
        }
    }
}

