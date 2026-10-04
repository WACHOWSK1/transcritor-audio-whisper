import os
import sys
import time
import argparse
from pathlib import Path
import ctranslate2
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
    output_formats: list = None,
    save_individual: bool = True
) -> dict:
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

    if save_individual:
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
            formatted_md_lines = "\n".join(markdown_lines)
            formatted_text_lines = "\n\n".join(text_only_lines)
            current_time = time.strftime('%d/%m/%Y %H:%M:%S')
            md_content = f"""# Transcrição: {stem}

- **Arquivo Original:** `{file_path.name}`
- **Duração:** {duration_str}
- **Idioma:** {info.language}
- **Processado em:** {current_time}

---

## 📝 Transcrição com Marcações de Tempo

{formatted_md_lines}

---

## 📄 Texto Contínuo

{formatted_text_lines}
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

    return {
        "file_path": file_path,
        "name": file_path.name,
        "stem": stem,
        "duration": info.duration,
        "duration_fmt": duration_str,
        "language": info.language,
        "language_prob": info.language_probability,
        "segments": segments_data,
        "output_lines": output_lines,
        "markdown_lines": markdown_lines,
        "text_only_lines": text_only_lines,
        "continuous_text": " ".join(text_only_lines),
        "saved_files": saved_files
    }

def save_merged_transcription(
    results: list,
    output_dir: Path,
    base_name: str = "transcricao_unificada",
    output_formats: list = None
) -> list:
    if output_dir is None:
        output_dir = results[0]["file_path"].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_formats is None:
        output_formats = ["txt", "md"]

    total_duration_sec = sum(r["duration"] for r in results)
    total_duration_str = format_timestamp(total_duration_sec)
    processed_time = time.strftime('%d/%m/%Y %H:%M:%S')
    saved_files = []

    # Markdown format
    if "md" in output_formats:
        md_path = output_dir / f"{base_name}.md"
        md_lines = [
            f"# 🎙️ Transcrição Consolidada",
            "",
            f"- **Total de Áudios:** {len(results)}",
            f"- **Duração Total:** {total_duration_str}",
            f"- **Processado em:** {processed_time}",
            f"- **Diretório:** `{output_dir}`",
            "",
            "---",
            "",
            "## 📖 Texto Contínuo Unificado",
            ""
        ]

        for idx, res in enumerate(results, 1):
            md_lines.append(f"### 🕒 Parte {idx}: `{res['name']}` ({res['duration_fmt']})")
            if res["text_only_lines"]:
                md_lines.append("\n\n".join(res["text_only_lines"]))
            else:
                md_lines.append("*(Nenhuma fala detectada neste trecho)*")
            md_lines.append("")

        md_lines.append("---")
        md_lines.append("")
        md_lines.append("## ⏱️ Transcrição Detalhada com Marcações de Tempo")
        md_lines.append("")

        for idx, res in enumerate(results, 1):
            md_lines.append(f"### 📁 [{idx}/{len(results)}] `{res['name']}` (Duração: {res['duration_fmt']})")
            if res["markdown_lines"]:
                md_lines.extend(res["markdown_lines"])
            else:
                md_lines.append("- *(Nenhuma fala detectada)*")
            md_lines.append("")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        saved_files.append(md_path)

    # TXT format
    if "txt" in output_formats:
        txt_path = output_dir / f"{base_name}.txt"
        txt_lines = [
            "=" * 70,
            "TRANSCRICAO CONSOLIDADA",
            f"Total de Áudios: {len(results)}",
            f"Duração Total: {total_duration_str}",
            f"Processado em: {processed_time}",
            f"Diretório: {output_dir}",
            "=" * 70,
            "",
            "--- TEXTO CONTINUO UNIFICADO ---",
            ""
        ]

        for idx, res in enumerate(results, 1):
            txt_lines.append(f"[{idx}] {res['name']} ({res['duration_fmt']}):")
            if res["text_only_lines"]:
                txt_lines.append("\n".join(res["text_only_lines"]))
            else:
                txt_lines.append("(Nenhuma fala detectada)")
            txt_lines.append("")

        txt_lines.append("=" * 70)
        txt_lines.append("--- TRANSCRICAO DETALHADA COM MARCACOES DE TEMPO ---")
        txt_lines.append("=" * 70)
        txt_lines.append("")

        for idx, res in enumerate(results, 1):
            txt_lines.append(f"Arquivo [{idx}/{len(results)}]: {res['name']} (Duração: {res['duration_fmt']})")
            if res["output_lines"]:
                txt_lines.extend(res["output_lines"])
            else:
                txt_lines.append("(Nenhuma fala detectada)")
            txt_lines.append("")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))
        saved_files.append(txt_path)

    print("\n" + "=" * 60)
    print("ARQUIVO(S) UNIFICADO(S) GERADO(S) COM SUCESSO:")
    for path in saved_files:
        print(f"  -> {path}")
    print("=" * 60 + "\n")

    return saved_files

def main():
    parser = argparse.ArgumentParser(
        description="Transcritor de áudio e reuniões de alta precisão com Faster-Whisper."
    )
    parser.add_argument("input", help="Caminho para o arquivo de áudio/vídeo ou pasta de arquivos.")
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large-v3"], help="Tamanho do modelo Whisper (padrão: small).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Dispositivo de execução (cpu ou cuda; padrão: cpu).")
    parser.add_argument("--compute_type", default="int8", help="Tipo de quantização (int8 para CPU, float16 para CUDA; padrão: int8).")
    parser.add_argument("--language", default="pt", help="Código de idioma (padrão: pt).")
    parser.add_argument("--output_dir", default=None, help="Diretório onde salvar as transcrições.")
    parser.add_argument("--format", default="all", choices=["all", "txt", "md", "srt"], help="Formato de saída desejado.")
    parser.add_argument("--merge", action="store_true", help="Mescla todas as transcrições da pasta em um único arquivo consolidado.")
    parser.add_argument("--merge-only", action="store_true", help="Gera apenas o arquivo único consolidado, sem criar arquivos individuais por áudio.")
    parser.add_argument("--merge-name", default="transcricao_unificada", help="Nome base do arquivo unificado gerado (padrão: transcricao_unificada).")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Erro: O caminho '{input_path}' não foi encontrado.")
        sys.exit(1)

    device = args.device
    compute_type = args.compute_type
    if device == "cuda" and compute_type == "int8":
        compute_type = "float16"

    output_formats = ["txt", "md", "srt"] if args.format == "all" else [args.format]
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Load model
    print(f"\nCarregando modelo Faster-Whisper '{args.model}' no dispositivo '{device}' ({compute_type})...")
    load_start = time.time()
    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    except Exception as e:
        if device == "cuda":
            print(f"Falha ao iniciar com CUDA: {e}. Tentando fallback para CPU...")
            device = "cpu"
            compute_type = "int8"
            model = WhisperModel(args.model, device=device, compute_type=compute_type)
        else:
            raise e
    print(f"Modelo carregado em {time.time() - load_start:.1f}s.")

    # Process file or directory
    if input_path.is_file():
        transcribe_audio(input_path, model, language=args.language, output_dir=output_dir, output_formats=output_formats)
    elif input_path.is_dir():
        files = sorted(
            [p for p in input_path.iterdir() if p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS],
            key=lambda x: x.name
        )
        if not files:
            print(f"Nenhum arquivo de áudio com extensões suportadas encontrado em: {input_path}")
            sys.exit(1)

        print(f"Encontrados {len(files)} arquivo(s) para transcrição em ordem cronológica:")
        for idx, f in enumerate(files, 1):
            print(f"  {idx}. {f.name}")

        results = []
        save_individual = not args.merge_only

        for f in files:
            res = transcribe_audio(
                f,
                model,
                language=args.language,
                output_dir=output_dir,
                output_formats=output_formats,
                save_individual=save_individual
            )
            results.append(res)

        if args.merge or args.merge_only:
            merged_formats = ["txt", "md"] if args.format == "all" else [args.format]
            # If srt was specified but not txt/md, fallback to md and txt for merged
            if "srt" in merged_formats and "txt" not in merged_formats and "md" not in merged_formats:
                merged_formats = ["md", "txt"]

            target_output_dir = output_dir if output_dir else input_path
            save_merged_transcription(
                results,
                output_dir=target_output_dir,
                base_name=args.merge_name,
                output_formats=merged_formats
            )

if __name__ == "__main__":
    main()

