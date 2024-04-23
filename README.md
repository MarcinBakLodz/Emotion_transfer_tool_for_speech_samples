## Environment Setup

```bash
conda create --name emotional_speech pytorch torchvision torchaudio comet_ml pytorch-lightning torchinfo natsort ffmpeg pysoundfile tqdm pytorch-cuda=12.1 -c pytorch -c nvidia -c comet_ml -c conda-forge
```

or

```bash
conda env create -f linux_env.yml

```
or

```bash
conda env create -f windows_env.yml
```
