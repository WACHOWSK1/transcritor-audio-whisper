# 🎙️ Transcritor de Áudio e Reuniões (Faster-Whisper)

Ferramenta CLI rápida, eficiente e local para transcrição de reuniões, entrevistas e áudios em múltiplos formatos (`.txt`, `.md` e legendas `.srt`), com detecção automática de voz (VAD) e marcadores de tempo detalhados.

Construído utilizando [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (reimplementação do Whisper da OpenAI em CTranslate2), que é até **4x mais rápido** e consome significativamente menos memória RAM/VRAM.

---

## 🚀 Funcionalidades

- **Múltiplos Formatos de Saída:**
  - `Markdown (.md)`: Formatado com cabeçalhos, tempo de início de cada fala e seção de texto contínuo.
  - `Texto Puro (.txt)`: Transcrição linha a linha com intervalos `[00:00 -> 00:15]`.
  - `Legendas (.srt)`: Padrão de legendas pronto para vídeos.
- **Suporte a Arquivo Único ou Pasta:** Transcreva um arquivo isolado ou processe em lote todos os áudios de um diretório.
- **VAD (Voice Activity Detection):** Filtro de silêncios integrado para evitar alucinações em trechos mudos.
- **Execução em CPU ou GPU (CUDA):** Suporta quantização `int8` otimizada para CPU e `float16` para GPU.
- **Detecção e Tradução/Transição de Idiomas:** Otimizado por padrão para Português (`pt`), com suporte a mais de 90 línguas.

---

## 📦 Instalação e Pré-requisitos

### 1. Pré-requisitos
- **Python 3.10+**
- **FFmpeg**: Necessário para decodificação de áudio (m4a, mp3, etc.).

### 2. Instalação das dependências
Clone o repositório e instale as dependências:

```bash
git clone https://github.com/WACHOWSK1/transcritor-audio-whisper.git
cd transcritor-audio-whisper
pip install -r requirements.txt
```

---

## 🛠️ Como Usar

### Transcrever um único arquivo de áudio:
```bash
python transcribe.py "caminho/do/seu/audio.m4a"
```

### Transcrever todos os áudios de uma pasta e unificar em um único arquivo:
```bash
python transcribe.py "caminho/para/pasta_audios/" --merge
```

### Gerar apenas o arquivo unificado (sem criar arquivos para cada áudio):
```bash
python transcribe.py "caminho/para/pasta_audios/" --merge-only --merge-name "sessao_completa"
```

### Escolher tamanho do modelo (`tiny`, `base`, `small`, `medium`, `large-v3`):
```bash
python transcribe.py "reuniao.mp3" --model small
```

### Salvar em uma pasta específica:
```bash
python transcribe.py "reuniao.m4a" --output_dir "saida_transcricoes"
```

### Gerar apenas um formato específico (`md`, `txt` ou `srt`):
```bash
python transcribe.py "video.mp4" --format srt
```

---

## ⚙️ Opções da Linha de Comando

| Argumento | Padrão | Descrição |
| :--- | :--- | :--- |
| `input` | *(obrigatório)* | Caminho do arquivo ou diretório de áudios. |
| `--model` | `small` | Tamanho do modelo Whisper (`tiny`, `base`, `small`, `medium`, `large-v3`). |
| `--device` | `cpu` | Dispositivo de execução (`cpu` ou `cuda`). |
| `--compute_type` | `int8` | Quantização (`int8` para CPU, `float16` para GPU CUDA). |
| `--language` | `pt` | Código do idioma (ex.: `pt`, `en`, `es`). |
| `--output_dir` | *(mesma pasta do áudio)* | Diretório de destino dos arquivos gerados. |
| `--format` | `all` | Formato gerado: `all`, `md`, `txt`, ou `srt`. |
| `--merge` | `False` | Mescla todos os áudios da pasta em um único arquivo consolidado. |
| `--merge-only` | `False` | Gera apenas o arquivo consolidado unificado (sem arquivos individuais). |
| `--merge-name` | `transcricao_unificada` | Nome base do arquivo unificado gerado. |

---

## 📄 Licença

Distribuído sob a licença MIT. Consulte `LICENSE` para mais detalhes.
