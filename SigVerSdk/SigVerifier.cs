using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace SigVerSdk;

public class SigVerifier : IDisposable
{
    private readonly InferenceSession _session;

    private const int CanvasWidth = 1360;
    private const int CanvasHeight = 840;
    private const int ResizeWidth = 242;
    private const int ResizeHeight = 170;

    static SigVerifier()
    {
        try
        {
            var libPath = System.IO.Path.Combine(AppContext.BaseDirectory, "libOpenCvSharpExtern.so");
            if (System.IO.File.Exists(libPath))
                NativeLibrary.Load(libPath);
        }
        catch
        {
            // ignore if loading fails, rely on system paths
        }
    }

    public SigVerifier(string modelPath)
    {
        _session = new InferenceSession(modelPath);
    }

    private static Mat Preprocess(Mat bitmap)
    {
        Mat gray;
        if (bitmap.Channels() == 1)
            gray = bitmap.Clone();
        else
        {
            gray = new Mat();
            Cv2.CvtColor(bitmap, gray, ColorConversionCodes.BGR2GRAY);
        }

        byte thr = (byte)Cv2.Threshold(gray, new Mat(), 0, 255, ThresholdTypes.Otsu);

        using var grayF = new Mat();
        gray.ConvertTo(grayF, MatType.CV_64F);
        using var blurred = new Mat();
        Cv2.GaussianBlur(grayF, blurred, new Size(17, 17), 2, 0, BorderTypes.Reflect101);

        int minR = blurred.Rows, maxR = -1, minC = blurred.Cols, maxC = -1;
        long sumR = 0, sumC = 0, count = 0;
        for (int y = 0; y < blurred.Rows; y++)
        {
            for (int x = 0; x < blurred.Cols; x++)
            {
                double val = blurred.At<double>(y, x);
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

        var subsetRect = new Rect(minC, minR, maxC - minC + 1, maxR - minR + 1);
        using var cropped = new Mat(grayF, subsetRect);

        var canvas = new Mat(CanvasHeight, CanvasWidth, MatType.CV_64F, Scalar.All(255));
        int rStart = CanvasHeight / 2 - rCenter;
        int cStart = CanvasWidth / 2 - cCenter;
        if (rStart < 0) rStart = 0;
        if (cStart < 0) cStart = 0;
        var roi = new Rect(cStart, rStart, Math.Min(cropped.Cols, CanvasWidth - cStart), Math.Min(cropped.Rows, CanvasHeight - rStart));
        var srcRoi = new Rect(0, 0, roi.Width, roi.Height);
        cropped[srcRoi].CopyTo(new Mat(canvas, roi));

        for (int y = 0; y < canvas.Rows; y++)
        {
            for (int x = 0; x < canvas.Cols; x++)
            {
                double v = canvas.At<double>(y, x);
                if (v > thr) v = 255;
                canvas.Set<double>(y, x, 255 - v);
            }
        }

        using var resized = new Mat();
        Cv2.Resize(canvas, resized, new Size(ResizeWidth, ResizeHeight), 0, 0, InterpolationFlags.Area);

        var cropRect = new Rect((ResizeWidth - 220) / 2, (ResizeHeight - 150) / 2, 220, 150);
        using var finalF = new Mat(resized, cropRect);
        var finalMat = new Mat();
        finalF.ConvertTo(finalMat, MatType.CV_8U);
        canvas.Dispose();
        gray.Dispose();
        return finalMat;
    }

    public void SavePreprocessed(string inputPath, string outputPath)
    {
        using var bmp = Cv2.ImRead(inputPath, ImreadModes.Color);
        if (bmp.Empty())
            throw new ArgumentException($"Unable to load image '{inputPath}'");
        using var pre = Preprocess(bmp);
        Cv2.ImWrite(outputPath, pre);
    }

    public float[] ExtractFeatures(string imagePath)
    {
        using var bitmap = Cv2.ImRead(imagePath, ImreadModes.Color);
        if (bitmap.Empty())
            throw new ArgumentException($"Unable to load image '{imagePath}'");
        using var pre = Preprocess(bitmap);
        var input = new DenseTensor<float>(new[] { 1, 1, 150, 220 });
        for (int y = 0; y < 150; y++)
        {
            for (int x = 0; x < 220; x++)
            {
                input[0, 0, y, x] = pre.At<byte>(y, x) / 255f;
            }
        }
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", input) };
        using var results = _session.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();
        if (output.Length != 2048 || output.Any(float.IsNaN) || output.Any(float.IsInfinity))
            throw new InvalidOperationException("Model returned invalid features");
        return output;
    }

    public static void Normalize(Span<float> v)
    {
        double n = 0;
        foreach (var x in v) n += x * x;
        n = Math.Sqrt(n);
        for (int i = 0; i < v.Length; i++) v[i] /= (float)n;
    }

    public static double CosineDistance(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double dot = 0;
        for (int i = 0; i < a.Length; i++) dot += a[i] * b[i];
        return 1 - dot;
    }

    public bool IsForgery(string referencePath, string candidatePath, float threshold = 0.35f)
    {
        var refFeatures = ExtractFeatures(referencePath);
        var candFeatures = ExtractFeatures(candidatePath);
        Normalize(refFeatures);
        Normalize(candFeatures);

        double distance = CosineDistance(refFeatures, candFeatures);
        return distance > threshold;
    }

    public void Dispose()
    {
        _session.Dispose();
    }
}
