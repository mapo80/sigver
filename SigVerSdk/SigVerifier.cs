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

    private static readonly SKImageInfo CanvasInfo = new(1360, 840, SKColorType.Gray8, SKAlphaType.Opaque);
    private const int ResizeWidth = 242;
    private const int ResizeHeight = 170;

    public SigVerifier(string modelPath)
    {
        _session = new InferenceSession(modelPath);
    }

    private static SKBitmap Preprocess(SKBitmap bitmap)
    {
        using var gray = new SKBitmap(bitmap.Width, bitmap.Height, SKColorType.Gray8, SKAlphaType.Opaque);
        for (int y = 0; y < bitmap.Height; y++)
        {
            for (int x = 0; x < bitmap.Width; x++)
            {
                var c = bitmap.GetPixel(x, y);
                byte l = (byte)(0.299f * c.Red + 0.587f * c.Green + 0.114f * c.Blue);
                gray.SetPixel(x, y, new SKColor(l, l, l));
            }
        }

        byte thr = OtsuThreshold(gray);

        using var blurred = GaussianBlur(gray, 2f);
        int minR = blurred.Height, maxR = -1, minC = blurred.Width, maxC = -1;
        long sumR = 0, sumC = 0, count = 0;
        for (int y = 0; y < blurred.Height; y++)
        {
            for (int x = 0; x < blurred.Width; x++)
            {
                byte val = blurred.GetPixel(x, y).Red;
                if (val <= thr)
                {
                    if (y < minR) minR = y;
                    if (y > maxR) maxR = y;
                    if (x < minC) minC = x;
                    if (x > maxC) maxC = x;
                    sumR += y;
                    sumC += x;
                    count++;
                }
            }
        }
        if (count == 0)
            throw new InvalidOperationException("Signature appears to be blank");

        int rCenter = (int)(sumR / (double)count) - minR;
        int cCenter = (int)(sumC / (double)count) - minC;

        var subsetRect = SKRectI.Create(minC, minR, maxC - minC + 1, maxR - minR + 1);
        using var cropped = new SKBitmap(subsetRect.Width, subsetRect.Height, SKColorType.Gray8, SKAlphaType.Opaque);
        gray.ExtractSubset(cropped, subsetRect);

        var canvas = new SKBitmap(CanvasInfo);
        canvas.Erase(SKColors.White);
        int rStart = CanvasInfo.Height / 2 - rCenter;
        int cStart = CanvasInfo.Width / 2 - cCenter;
        if (rStart < 0) rStart = 0;
        if (cStart < 0) cStart = 0;
        using (var cv = new SKCanvas(canvas))
            cv.DrawBitmap(cropped, cStart, rStart);

        for (int y = 0; y < canvas.Height; y++)
        {
            for (int x = 0; x < canvas.Width; x++)
            {
                var v = canvas.GetPixel(x, y).Red;
                if (v > thr) v = 255;
                canvas.SetPixel(x, y, new SKColor((byte)(255 - v), (byte)(255 - v), (byte)(255 - v)));
            }
        }

        using var resized = canvas.Resize(new SKImageInfo(ResizeWidth, ResizeHeight), SKFilterQuality.High)
            ?? throw new InvalidOperationException("Failed to resize image");
        var cropRect = SKRectI.Create((ResizeWidth - 220) / 2, (ResizeHeight - 150) / 2, 220, 150);
        var finalBmp = new SKBitmap(cropRect.Width, cropRect.Height, SKColorType.Gray8, SKAlphaType.Opaque);
        resized.ExtractSubset(finalBmp, cropRect);
        return finalBmp;
    }

    /// <summary>
    /// Saves the preprocessed grayscale image used for feature extraction.
    /// </summary>
    /// <param name="inputPath">Path to the original image.</param>
    /// <param name="outputPath">Where to save the processed 150x220 image.</param>
    public void SavePreprocessed(string inputPath, string outputPath)
    {
        using var bmp = SKBitmap.Decode(inputPath) ??
            throw new ArgumentException($"Unable to load image '{inputPath}'");
        using var pre = Preprocess(bmp);
        using var image = SKImage.FromBitmap(pre);
        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        using var fs = System.IO.File.OpenWrite(outputPath);
        data.SaveTo(fs);
    }

    private static byte OtsuThreshold(SKBitmap gray)
    {
        long[] hist = new long[256];
        for (int y = 0; y < gray.Height; y++)
        {
            for (int x = 0; x < gray.Width; x++)
            {
                hist[gray.GetPixel(x, y).Red]++;
            }
        }
        int total = gray.Width * gray.Height;
        double sum = 0;
        for (int t = 0; t < 256; t++) sum += t * hist[t];
        double sumB = 0, wB = 0, wF = 0, varMax = 0;
        int threshold = 0;
        for (int t = 0; t < 256; t++)
        {
            wB += hist[t];
            if (wB == 0) continue;
            wF = total - wB;
            if (wF == 0) break;
            sumB += t * hist[t];
            double mB = sumB / wB;
            double mF = (sum - sumB) / wF;
            double varBetween = wB * wF * (mB - mF) * (mB - mF);
            if (varBetween > varMax)
            {
                varMax = varBetween;
                threshold = t;
            }
        }
        return (byte)threshold;
    }

    private static SKBitmap GaussianBlur(SKBitmap src, float sigma)
    {
        var info = new SKImageInfo(src.Width, src.Height, src.ColorType, src.AlphaType);
        using var surface = SKSurface.Create(info);
        var paint = new SKPaint { ImageFilter = SKImageFilter.CreateBlur(sigma, sigma, SKShaderTileMode.Clamp) };
        surface.Canvas.DrawBitmap(src, 0, 0, paint);
        surface.Canvas.Flush();
        using var image = surface.Snapshot();
        var dst = new SKBitmap(info);
        image.ReadPixels(dst.Info, dst.GetPixels(), dst.Info.RowBytes, 0, 0);
        return dst;
    }

    public float[] ExtractFeatures(string imagePath)
    {
        using var bitmap = SKBitmap.Decode(imagePath) ??
            throw new ArgumentException($"Unable to load image '{imagePath}'");
        using var pre = Preprocess(bitmap);
        var input = new DenseTensor<float>(new[] { 1, 1, 150, 220 });
        for (int y = 0; y < 150; y++)
        {
            for (int x = 0; x < 220; x++)
            {
                input[0, 0, y, x] = pre.GetPixel(x, y).Red / 255f;
            }
        }
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", input) };
        using var results = _session.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();
        if (output.Length != 2048 || output.Any(float.IsNaN) || output.Any(float.IsInfinity))
            throw new InvalidOperationException("Model returned invalid features");
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
