```python
# -*- coding: utf-8 -*-

# ==================================================================================================
# =                                                                                              =
# =                                     自动时间轴脚本                                         =
# =                                                                                              =
# ==================================================================================================
#
# 描述:
# 本脚本旨在自动化为视频“打轴”（字幕同步）的过程。
# 它接收一个视频文件和一个不带时间戳的字幕文件作为输入，
# 并生成一个时间完美对齐的新的 .ass 字幕文件。
#
# 工作流程:
# 1. 使用 FFmpeg 从输入视频中提取音轨。
# 2. 使用 Demucs 模型从音轨中分离人声，以获得更纯净的音频。
# 3. 使用 Faster-Whisper 转录人声，生成带有词级时间戳的文本。
# 4. 读取用户提供的字幕行。
# 5. 使用序列匹配算法，将用户的每一行字幕与转录的词语进行对齐。
# 6. 调整时间戳以获得更好的视觉节奏（增加少量缓冲）。
# 7. 生成包含文本和新时间戳的最终 .ass 字幕文件。
#
# ==================================================================================================
#
# 安装与环境配置指南:
#
# 1. 安装 FFmpeg:
#    本脚本要求系统中已安装 FFmpeg 并将其添加至系统路径 (PATH)。
#    - Windows: 从 https://ffmpeg.org/download.html 下载，并将 'bin' 文件夹添加到 PATH 环境变量中。
#    - macOS (使用 Homebrew): brew install ffmpeg
#    - Linux (使用 apt): sudo apt update && sudo apt install ffmpeg
#
# 2. 安装 PYTHON 和 PIP:
#    请确保已安装 Python 3.8 或更高版本。
#
# 3. 创建虚拟环境 (推荐):
#    python -m venv venv
#    source venv/bin/activate  # 在 Linux/macOS 上
#    .\venv\Scripts\activate   # 在 Windows 上
#
# 4. 安装 PYTHON 依赖项:
#    运行以下命令安装所有必需的库。为获得更佳性能，强烈建议使用 NVIDIA GPU，
#    请先安装支持 CUDA 的 PyTorch。
#
#    # 主要安装命令 (CPU 或已安装带 CUDA 支持的 PyTorch 的 GPU):
#    pip install faster-whisper demucs pysubs2 "thefuzz[speedup]" torch torchaudio --upgrade
#
#    # GPU 注意事项: 为获得显著的速度提升，请在运行上述命令前，
#    # 根据您的系统安装支持 CUDA 的 PyTorch。访问 https://pytorch.org/get-started/locally/
#    # 查找适合您配置 (CUDA/ROCm) 的正确命令。
#
# ==================================================================================================
#
# 执行方法:
#
# 从命令行运行此脚本，并提供视频文件和字幕文件的路径。
#
# 命令示例:
# python auto_timeline_script.py --video "路径/到/你的/视频.mp4" --subtitles "路径/到/你的/字幕.ass"
#
# 可选参数:
# --output [文件路径]   : 指定输出路径。默认: [视频文件名]_timed.ass
# --model [模型大小]      : Whisper 模型大小 (tiny, base, small, medium, large-v3)。默认: medium。
# --device [设备]        : 使用的设备 (cuda, cpu)。默认: 如果可用，自动检测 cuda。
# --buffer [秒]           : 添加到每行字幕开头/结尾的缓冲时间。默认: 0.05
#
# ==================================================================================================

import argparse
import os
import subprocess
import sys
import tempfile
import shutil
import re
from pathlib import Path
from typing import List, Dict, Any

try:
    import torch
    import torchaudio
    from demucs.separate import main as demucs_main
    from faster_whisper import WhisperModel
    import pysubs2
    from thefuzz import fuzz
except ImportError as e:
    print(f"导入错误: {e}。请确保所有依赖项都已安装。")
    print("请运行: pip install faster-whisper demucs pysubs2 \"thefuzz[speedup]\" torch torchaudio")
    sys.exit(1)


def check_dependencies():
    """检查外部依赖项 (FFmpeg) 是否已安装并存在于系统路径中。"""
    print("正在检查依赖项...")
    if not shutil.which("ffmpeg"):
        print("\x1b[91m错误: 未找到 FFmpeg。\x1b[0m")
        print("FFmpeg 是从视频文件中提取音频所必需的。")
        print("请安装它并确保它在您的系统路径中。")
        print("下载地址: https://ffmpeg.org/download.html")
        sys.exit(1)
    print("--> 已找到 FFmpeg。")


def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """
    使用 FFmpeg 从视频文件中提取音轨。

    音频被转换为 16kHz 单声道 WAV 格式，这是 Whisper 的理想格式。
    """
    print(f"\n正在从 '{video_path.name}' 提取音频...")
    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",          # 无视频
        "-ac", "1",     # 单声道音频
        "-ar", "16000", # 16kHz 采样率
        "-c:a", "pcm_s16le", # WAV 格式
        "-y",           # 如果输出文件已存在则覆盖
        str(audio_path),
    ]
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"--> 成功提取音频至 '{audio_path.name}'。")
        return True
    except subprocess.CalledProcessError as e:
        print("\x1b[91m执行 FFmpeg 时出错:\x1b[0m")
        print(e.stderr)
        return False


def separate_vocals(audio_path: Path, output_dir: Path, device: str) -> Path:
    """
    使用 Demucs 从音轨中分离人声。

    这能显著提高在有背景音乐或噪音的视频中进行语音识别的准确性。
    """
    print("\n正在从音频中分离人声 (此过程可能需要一些时间)...")
    model_name = "htdemucs_ft"
    demucs_args = [
        "-d", device,
        "-n", model_name,
        "--two-stems=vocals",
        "-o", str(output_dir),
        str(audio_path),
    ]
    
    # Demucs API 需要一个字符串列表
    demucs_main(demucs_args)
    
    # Demucs 会创建一个以模型名称命名的子文件夹
    vocals_file = output_dir / model_name / audio_path.stem / "vocals.wav"
    
    if not vocals_file.exists():
        raise FileNotFoundError(f"预期的人声文件未在 '{vocals_file}' 找到")
    
    print(f"--> 人声分离成功: '{vocals_file}'。")
    return vocals_file


def transcribe_with_timestamps(vocals_path: Path, model_size: str, device: str) -> List[Dict[str, Any]]:
    """
    使用 Faster-Whisper 转录人声音频，以获得词级时间戳。
    """
    print(f"\n正在使用 Whisper '{model_size}' 模型转录人声...")
    compute_type = "float16" if device == "cuda" else "int8"
    
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"\x1b[91m加载 Whisper 模型时出错: {e}\x1b[0m")
        print("如果您正在使用 GPU，请检查您的 PyTorch 安装和 CUDA 兼容性。")
        sys.exit(1)
        
    segments, _ = model.transcribe(
        str(vocals_path),
        word_timestamps=True,
        language="pt" # 将语言设置为葡萄牙语
    )

    all_words = []
    for segment in segments:
        for word in segment.words:
            all_words.append({
                "word": word.word,
                "start": word.start,
                "end": word.end
            })

    print(f"--> 转录完成。识别出 {len(all_words)} 个单词。")
    return all_words


def normalize_text(text: str) -> str:
    """标准化文本以进行更稳健的匹配。"""
    text = text.lower().strip()
    # 移除大部分标点符号，保留内部的撇号和连字符
    text = re.sub(r"[.,!?;:\"“”()\[\]]", "", text)
    # 将多个空格替换为单个空格
    text = re.sub(r"\s+", " ", text)
    return text


def align_subtitles(
    subtitle_lines: List[str],
    whisper_words: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    用于将用户字幕行与转录词语对齐的核心算法。

    它使用一种带模糊匹配的贪心序列匹配算法，以保证稳健性。
    """
    print("\n正在将字幕与转录文本对齐...")
    
    timed_events = []
    whisper_word_idx = 0
    
    for i, line in enumerate(subtitle_lines):
        if not line.strip():
            continue

        normalized_line = normalize_text(line)
        line_words = normalized_line.split()
        
        if not line_words:
            continue

        best_match = {
            "score": -1,
            "start_idx": -1,
            "end_idx": -1
        }
        
        # 优化: 估算 Whisper 词语序列的长度
        avg_word_len = sum(len(w['word']) for w in whisper_words[whisper_word_idx:whisper_word_idx+50]) / 50 if whisper_word_idx + 50 < len(whisper_words) else 5
        whisper_len_estimate = int(len(normalized_line) / avg_word_len * 1.5) + len(line_words)
        search_end_idx = min(len(whisper_words), whisper_word_idx + whisper_len_estimate)

        # 遍历可能的 Whisper 词语序列
        for start_idx in range(whisper_word_idx, search_end_idx):
            for end_idx in range(start_idx, search_end_idx):
                # 从 Whisper 词语构建候选句子
                candidate_words = whisper_words[start_idx : end_idx + 1]
                candidate_text = normalize_text(" ".join(w['word'] for w in candidate_words))
                
                if not candidate_text:
                    continue

                # 计算相似度分数
                score = fuzz.ratio(normalized_line, candidate_text)

                # 惩罚较大的长度差异以避免无意义的匹配
                len_diff_penalty = 1 - (abs(len(line_words) - len(candidate_words)) / len(line_words))
                if len_diff_penalty < 0: len_diff_penalty = 0
                
                adjusted_score = score * len_diff_penalty

                if adjusted_score > best_match["score"]:
                    best_match["score"] = adjusted_score
                    best_match["start_idx"] = start_idx
                    best_match["end_idx"] = end_idx

        # 如果找到足够好的匹配，则创建事件
        # 70 的阈值是一个很好的起点，可根据需要进行调整
        if best_match["score"] > 70 and best_match["start_idx"] != -1:
            start_word = whisper_words[best_match["start_idx"]]
            end_word = whisper_words[best_match["end_idx"]]
            
            event = {
                "text": line,
                "start": start_word['start'],
                "end": end_word['end'],
                "score": best_match['score']
            }
            timed_events.append(event)
            
            # 将 Whisper 词语索引前移到匹配项之后
            whisper_word_idx = best_match["end_idx"] + 1
            print(f"  [成功] 第 {i+1:03d} 行匹配成功，得分 {best_match['score']:.1f}")
        else:
            print(f"  \x1b[93m[警告] 第 {i+1:03d} 行未能找到良好匹配 (最佳得分: {best_match['score']:.1f})。已跳过。\x1b[0m")

    print(f"--> 对齐完成。{len(timed_events)}/{len(subtitle_lines)} 行字幕已同步。")
    return timed_events


def create_output_file(
    timed_events: List[Dict[str, Any]],
    output_path: Path,
    buffer: float
) -> None:
    """
    使用 pysubs2 创建最终的 .ass 字幕文件，并应用时间调整。
    """
    print(f"\n正在创建输出文件于 '{output_path}'...")
    
    subs = pysubs2.SSAFile()
    
    for i, event in enumerate(timed_events):
        start_time = event['start']
        end_time = event['end']
        
        # 添加时间缓冲，除非该字幕与前一条是连续的
        is_continuous = False
        if i > 0:
            prev_event = timed_events[i-1]
            # 如果时间差小于 200ms，则视为连续
            if start_time - prev_event['end'] < 0.2:
                is_continuous = True

        if not is_continuous:
            start_time = max(0, start_time - buffer)
            end_time += buffer
            
        ssa_event = pysubs2.SSAEvent(
            start=pysubs2.make_time(s=start_time),
            end=pysubs2.make_time(s=end_time),
            text=event['text'].replace(r'\N', '\\N') # 确保换行符 (\N) 被正确处理
        )
        subs.append(ssa_event)
        
    subs.save(str(output_path), encoding="utf-8-sig")
    print(f"--> 带时间轴的字幕文件保存成功。")


def main():
    """主函数，负责协调整个流程。"""
    parser = argparse.ArgumentParser(description="自动为视频同步字幕时间轴。")
    parser.add_argument("--video", type=Path, required=True, help="输入视频文件的路径。")
    parser.add_argument("--subtitles", type=Path, required=True, help="无时间轴的字幕文件路径 (.ass, .srt)。")
    parser.add_argument("--output", type=Path, help="输出的 .ass 文件路径。默认: [视频文件名]_timed.ass")
    parser.add_argument("--model", default="medium", choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"], help="要使用的 Whisper 模型大小。")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"], help="用于计算的设备 ('cuda' 或 'cpu')。默认: 自动检测。")
    parser.add_argument("--buffer", type=float, default=0.05, help="添加到每条字幕开头/结尾的缓冲时间(秒)。")
    
    args = parser.parse_args()

    # --- 输入验证 ---
    if not args.video.is_file():
        print(f"\x1b[91m错误: 在 '{args.video}' 未找到视频文件\x1b[0m")
        sys.exit(1)
    if not args.subtitles.is_file():
        print(f"\x1b[91m错误: 在 '{args.subtitles}' 未找到字幕文件\x1b[0m")
        sys.exit(1)
        
    output_path = args.output or args.video.with_name(f"{args.video.stem}_timed.ass")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("      开始自动字幕同步流程")
    print("=" * 60)
    print(f"输入视频         : {args.video}")
    print(f"输入字幕         : {args.subtitles}")
    print(f"输出文件         : {output_path}")
    print(f"计算设备         : {device.upper()}")
    print(f"Whisper 模型     : {args.model}")
    print("-" * 60)
    
    # 为中间文件创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="auto_timeline_")
    
    try:
        # --- 步骤 1: 检查依赖项 ---
        check_dependencies()
        
        # --- 步骤 2: 提取音频 ---
        temp_audio_path = Path(temp_dir) / "extracted_audio.wav"
        if not extract_audio(args.video, temp_audio_path):
            raise RuntimeError("音频提取失败。")

        # --- 步骤 3: 分离人声 ---
        demucs_output_dir = Path(temp_dir) / "demucs_output"
        demucs_output_dir.mkdir()
        vocals_path = separate_vocals(temp_audio_path, demucs_output_dir, device)

        # --- 步骤 4: 转录 ---
        whisper_words = transcribe_with_timestamps(vocals_path, args.model, device)
        if not whisper_words:
            raise RuntimeError("转录未产生任何词语。请检查音轨。")

        # --- 步骤 5: 读取用户字幕 ---
        subs_in = pysubs2.load(str(args.subtitles))
        subtitle_lines = [event.text for event in subs_in if not event.is_comment]

        # --- 步骤 6: 对齐 ---
        timed_events = align_subtitles(subtitle_lines, whisper_words)
        if not timed_events:
            raise RuntimeError("对齐失败。无法同步任何字幕。")
            
        # --- 步骤 7: 创建输出文件 ---
        create_output_file(timed_events, output_path, args.buffer)
        
        print("\n" + "="*60)
        print("\x1b[92m流程成功完成！\x1b[0m")
        print(f"已同步的字幕文件已保存至: \x1b[1m{output_path}\x1b[0m")
        print("="*60)

    except (RuntimeError, FileNotFoundError) as e:
        print("\n" + "="*60)
        print(f"\x1b[91m发生严重错误:\x1b[0m {e}")
        print("流程已中止。")
        print("="*60)
        sys.exit(1)
        
    finally:
        # --- 清理 ---
        print("\n正在清理临时文件...")
        shutil.rmtree(temp_dir)
        print("--> 清理完成。")


if __name__ == "__main__":
    main()
```
