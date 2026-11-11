# Optional: Parakeet speech-to-text for a voice chat

Not part of the recipe's start. Run it only if you want to talk to the model: it puts NVIDIA's
[Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (25 European languages, CC BY 4.0) on one
of the Sparks behind an OpenAI-compatible `POST /v1/audio/transcriptions`, as a systemd user service. A voice chat
then is three calls: your audio to this endpoint, its text to the GLM server's `/v1/chat/completions`, and the reply
to a text-to-speech of your choice (none is included here).

```bash
./vince/extras/parakeet/install.sh                               # on the recipe's first worker, 127.0.0.1:8189
PARAKEET_HOST=user@spark3 ./vince/extras/parakeet/install.sh     # on the Spark you name, over ssh
./vince/extras/parakeet/install.sh status | stop | start | uninstall

# on that Spark:
curl -s http://127.0.0.1:8189/v1/audio/transcriptions -F file=@question.wav
# {"text":"..."}
```

**It does not go on the head.** With no target named, the installer takes the first worker from `scripts/local.sh`:
the head carries rank 0 and the API, runs hottest, and is where your other tools live. `PARAKEET_HOST=local` installs
on the machine you are on, if you really want that.

Any client that speaks OpenAI's transcription API works (the model also answers to the name `whisper-1`); the file
can be wav, mp3, m4a, ogg, webm or anything else FFmpeg reads.

## Settings

| Variable | Default | What it sets |
|---|---|---|
| `PARAKEET_HOST` | `WORKER` from `scripts/local.sh` | ssh target of the Spark to install on (`local`: this machine) |
| `PARAKEET_HOME` | `~/parakeet-stt` | where the environment and the model live (about 8 GB on disk) |
| `PARAKEET_PORT` | `8189` | port |
| `PARAKEET_BIND` | `127.0.0.1` | address it listens on |
| `PARAKEET_API_KEY` | none | if set, every transcription must send `Authorization: Bearer <key>` |

## Before you open it to a network

By default only that Spark itself can reach it. The head, a phone or a voice app elsewhere needs `PARAKEET_BIND` set
to the Spark's address on that network. The audio you send is speech, often about other people: set `PARAKEET_API_KEY` and keep
the port behind your firewall or VPN, and put TLS in front of it if the network is not yours. The server keeps no
audio and no text: its log records only each request's duration and length.

## What it costs

It holds about 4.5 GiB of the Spark's memory here, beside a GLM rank: 2.9 GiB for the model on the GPU and 1.5 GiB
of process memory. The installer refuses to start with under 8 GiB available; after a v2.4 start the Sparks here had
18-22 GiB. Speed: 6 s of speech in 0.17 s.

## What was checked, and what was not

On one Spark of this cluster: install (with the packages and the model already on disk), start, the key (401 without
it), English and Italian clips, status, uninstall. The server loads in 1-2 s; without the copy described in
`server.py` it took over 6 minutes on DGX OS driver 615. **Not replayed:** an install from an empty machine, where
`pip` downloads the packages listed in `requirements.txt` (the versions this ran with) and the installer downloads
the model from Hugging Face.

## Credits

Parakeet TDT 0.6B v3 is NVIDIA's, under CC BY 4.0; it is downloaded from Hugging Face and not part of this
repository. The server uses Hugging Face Transformers (Apache-2.0), PyTorch (BSD-3-Clause), PyAV (BSD-3-Clause,
with FFmpeg) and FastAPI (MIT).
