using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace SigVerSdk;

public class SigVerifier : IDisposable
{
    private readonly InferenceSession _session;

    public SigVerifier(string modelPath)
    {
        _session = new InferenceSession(modelPath);
    }

    public float[] ExtractFeatures(string imagePath)
    {
        using var bitmap = SKBitmap.Decode(imagePath) ?? throw new ArgumentException($"Unable to load image '{imagePath}'");
        using var resized = bitmap.Resize(new SKImageInfo(220, 150), SKFilterQuality.Medium)
            ?? throw new InvalidOperationException("Failed to resize image");
        var input = new DenseTensor<float>(new[] { 1, 1, 150, 220 });
        for (int y = 0; y < 150; y++)
        {
            for (int x = 0; x < 220; x++)
            {
                var color = resized.GetPixel(x, y);
                float luminance = 0.299f * color.Red + 0.587f * color.Green + 0.114f * color.Blue;
                input[0, 0, y, x] = luminance / 255f;
            }
        }
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", input) };
        using var results = _session.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();
        return output;
    }

    /// <summary>
    /// Compares two signatures and returns true when the candidate is
    /// considered a forgery of the reference.
    /// </summary>
    /// <param name="referencePath">Path to a genuine reference signature.</param>
    /// <param name="candidatePath">Path to the signature to verify.</param>
    /// <param name="threshold">Distance threshold used to decide if the signature is forged.</param>
    public bool IsForgery(string referencePath, string candidatePath, float threshold = 1.5f)
    {
        var refFeatures = ExtractFeatures(referencePath);
        var candFeatures = ExtractFeatures(candidatePath);
        double sum = 0.0;
        for (int i = 0; i < refFeatures.Length; i++)
        {
            double diff = refFeatures[i] - candFeatures[i];
            sum += diff * diff;
        }
        var distance = Math.Sqrt(sum);
        return distance > threshold;
    }

    public void Dispose()
    {
        _session.Dispose();
    }
}
