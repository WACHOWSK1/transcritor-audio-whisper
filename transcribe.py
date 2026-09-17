import os
import sys
import time
import argparse
from pathlib import Path
from faster_whisper import WhisperModel

SUPPORTED_AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".ogg", ".flac", ".aac", ".wma", ".mp4", ".mkv", ".avi", ".mov"}

def format_timestamp(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

def format_srt_timestamp(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

def transcribe_audio(
    file_path: Path,
    model: WhisperModel,
    language: str = "pt",
    output_dir: Path = None,
    output_formats: list = None
):
    if output_dir is None:
        output_dir = file_path.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    if output_formats is None:
        output_formats = ["txt", "md", "srt"]

    print(f"\n{'='*60}")
    print(f"Processando arquivo: {file_path.name}")
    print(f"Tamanho: {file_path.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"{'='*60}\n")

    start_time = time.time()

    segments, info = model.transcribe(
        str(file_path),
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    duration_str = format_timestamp(info.duration)
    print(f"Idioma detectado: {info.language} (confiança: {info.language_probability:.2%})")
    print(f"Duração do áudio: {duration_str}")
    print("-" * 60)

    segments_data = []
    output_lines = []
    text_only_lines = []
    markdown_lines = []
    srt_blocks = []

    segment_idx = 1
    for seg in segments:
        start_fmt = format_timestamp(seg.start)
        end_fmt = format_timestamp(seg.end)
        text = seg.text.strip()
        if not text:
            continue

        print(f"[{start_fmt} -> {end_fmt}] {text}", flush=True)

        segments_data.append({
            "start": seg.start,
            "end": seg.end,
            "start_fmt": start_fmt,
            "end_fmt": end_fmt,
            "text": text
        })

        output_lines.append(f"[{start_fmt} -> {end_fmt}] {text}")
        text_only_lines.append(text)
        markdown_lines.append(f"- **`{start_fmt}`** {text}")

        # SRT format
        srt_start = format_srt_timestamp(seg.start)
        srt_end = format_srt_timestamp(seg.end)
        srt_blocks.append(f"{segment_idx}\n{srt_start} --> {srt_end}\n{text}\n")
        segment_idx += 1

    elapsed_time = time.time() - start_time
    print("-" * 60)
    print(f"Finalizado em {elapsed_time / 60:.2f} minutos ({elapsed_time:.1f} segundos).")

    stem = file_path.stem
    saved_files = []

    # TXT output
    if "txt" in output_formats:
        txt_path = output_dir / f"{stem}_transcricao.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Arquivo: {file_path.name}\nDuração: {duration_str}\n\n")
            f.write("\n".join(output_lines))
        saved_files.append(txt_path)

    # Markdown output
    if "md" in output_formats:
        md_path = output_dir / f"{stem}_transcricao.md"
        md_content = f"""# Transcrição: {stem}

- **Arquivo Original:** `{file_path.name}`
- **Duração:** {duration_str}
- **Idioma:** {info.language}
- **Processado em:** {time.strftime('%d/%m/%Y %H:%M:%S')}

---

## 📝 Transcrição com Marcações de Tempo

{"\n".join(markdown_lines)}

---

## 📄 Texto Contínuo

{"\n\n".join(text_only_lines)}
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        saved_files.append(md_path)

    # SRT Subtitle output
    if "srt" in output_formats:
        srt_path = output_dir / f"{stem}.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_blocks))
        saved_files.append(srt_path)

    print("\nArquivos salvos:")
    for path in saved_files:
        print(f"  -> {path}")

    return saved_files

def main():
    parser = argparse.ArgumentParser(
        description="Transcritor de áudio e reuniões de alta precisão com Faster-Whisper."
    )
    parser.add_argument("input", help="Caminho para o arquivo de áudio/vídeo ou pasta de arquivos.")
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large-v3"], help="Tamanho do modelo Whisper (padrão: small).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Dispositivo de execução (cpu ou cuda).")
    parser.add_argument("--compute_type", default="int8", help="Tipo de quantização (int8 para CPU, float16 para CUDA).")
    parser.add_argument("--language", default="pt", help="Código de idioma (padrão: pt).")
    parser.add_argument("--output_dir", default=None, help="Diretório onde salvar as transcrições.")
    parser.add_argument("--format", default="all", choices=["all", "txt", "md", "srt"], help="Formato de saída desejado.")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Erro: O caminho '{input_path}' não foi encontrado.")
        sys.exit(1)

    output_formats = ["txt", "md", "srt"] if args.format == "all" else [args.format]
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Load model
    print(f"\nCarregando modelo Faster-Whisper '{args.model}' no dispositivo '{args.device}' ({args.compute_type})...")
    load_start = time.time()
    try:
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    except Exception as e:
        if args.device == "cuda":
            print(f"Falha ao iniciar com CUDA: {e}. Tentando fallback para CPU...")
            model = WhisperModel(args.model, device="cpu", compute_type="int8")
        else:
            raise e
    print(f"Modelo carregado em {time.time() - load_start:.1f}s.")

    # Process file or directory
    if input_path.is_file():
        transcribe_audio(input_path, model, language=args.language, output_dir=output_dir, output_formats=output_formats)
    elif input_path.is_dir():
        files = [p for p in input_path.iterdir() if p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS]
        if not files:
            print(f"Nenhum arquivo de áudio com extensões suportadas encontrado em: {input_path}")
            sys.exit(1)
        print(f"Encontrados {len(files)} arquivo(s) para transcrição.")
        for f in files:
            transcribe_audio(f, model, language=args.language, output_dir=output_dir, output_formats=output_formats)

if __name__ == "__main__":
    main()
