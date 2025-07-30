using System;

namespace SigVerSdk;

public sealed class EnsembleVerifier : IDisposable
{
    private readonly SigVerifier _signet;
    private readonly SigVerifier _signetF;

    public EnsembleVerifier(string signetPath, string signetFPath)
    {
        _signet = new SigVerifier(signetPath);
        _signetF = new SigVerifier(signetFPath);
    }

    public bool IsForgery(string referencePath, string candidatePath, float temperature = 1.008f, float threshold = 0.0010f)
    {
        var fA1 = _signet.ExtractFeatures(referencePath);
        var fB1 = _signet.ExtractFeatures(candidatePath);
        SigVerifier.Normalize(fA1);
        SigVerifier.Normalize(fB1);
        float d1 = (float)SigVerifier.CosineDistance(fA1, fB1);

        var fA2 = _signetF.ExtractFeatures(referencePath);
        var fB2 = _signetF.ExtractFeatures(candidatePath);
        SigVerifier.Normalize(fA2);
        SigVerifier.Normalize(fB2);
        float d2 = (float)SigVerifier.CosineDistance(fA2, fB2);

        float sMin = Math.Min(d1, d2);
        float sCal = sMin / temperature;
        return sCal > threshold;
    }

    public void Dispose()
    {
        _signet.Dispose();
        _signetF.Dispose();
    }
}
