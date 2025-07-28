# Repository notes

This project was updated to work with Python 3.10 or newer.

## .NET SDK
If the .NET SDK is not available on your machine, you can install it locally
using the `dotnet-install.sh` script included in the repository:

```bash
./dotnet-install.sh --channel 8.0
```
This downloads a recent SDK to `~/.dotnet` which is then used by the C#
projects.

Il file so si trova nella directory so/libOpenCvSharpExtern.so ed è stato compilato su Ubuntu 24.04.
I progetti .NET SigVerSdk e PreprocessDump usano OpenCvSharp al posto di
SkiaSharp. Non installare il runtime di OpenCvSharp da NuGet: la libreria nativa
viene fornita in `so` e copiata automaticamente durante la compilazione.

Per eseguire i test è necessario installare l'SDK .NET 9 incluso nel repository:

```
./dotnet-install.sh --version 9.0.100 --install-dir "$HOME/dotnet"
export PATH="$HOME/dotnet:$PATH"
```

OpenCvSharp richiede che la libreria nativa sia accessibile. Aggiungi la cartella `so` al `LD_LIBRARY_PATH` oppure copia `libOpenCvSharpExtern.so` nella directory di output dei test:

```
export LD_LIBRARY_PATH="$PWD/so:$LD_LIBRARY_PATH"
```

Su sistemi appena installati potrebbero mancare alcune dipendenze di `libOpenCvSharpExtern.so`. Installa i pacchetti necessari con:

```
sudo apt-get update
sudo apt-get install -y libgtk2.0-0 libgdk-pixbuf2.0-0 libtesseract5 libdc1394-25 libavcodec60 libavformat60 libswscale7 libsm6 libxext6 libxrender1 libgomp1
```

Per verificare il corretto funzionamento di OpenCvSharp è presente il progetto di test `TestOpenCvSharp`. Eseguire i test con:

```
dotnet test TestOpenCvSharp/TestOpenCvSharp.csproj -v n
```
