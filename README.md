# Veau-sync-sub

# 自动视频字幕时间轴工具
## 📖 项目简介
本工具是一个自动化的命令行脚本，旨在解决为视频“打轴”（即同步字幕时间轴）的繁琐工作。您只需提供一个视频文件和一个不含时间信息的纯文本字幕文件，脚本即可通过语音识别和智能对齐，生成一个时间码精确匹配的 .ass 格式字幕文件。
它的核心工作流程如下：
  音频提取: 从视频中分离出音轨。
  人声分离: 使用 Demucs 模型去除背景音乐和噪音，提取纯净的人声。
  语音转录: 利用 Faster-Whisper 模型对人声进行高精度语音识别，生成带有词级时间戳的文本。
  智能对齐: 通过模糊字符串匹配算法，将您提供的每一行字幕与识别出的词语序列进行精确对齐。
  生成字幕: 结合对齐后的时间戳和原始字幕文本，生成一个全新的、时间精准的 .ass 字幕文件。
## ✨ 功能特性
  • 🤖 全自动处理: 从视频到最终的带时间轴字幕，全程无需手动干预。
  • 🎤 高质量人声分离: 集成 Demucs 模型，有效处理含背景音乐或噪音的视频，显著提升语音识别准确率。
  • 🎯 高精度语音识别: 采用 Faster-Whisper，支持多种模型尺寸，可根据需求在速度和精度之间进行权衡。
  • 🔍 智能文本对齐: 使用模糊匹配算法，即使字幕文本与实际语音有微小出入也能成功对齐。
  • 🎞️ ASS格式输出: 生成功能丰富的 .ass 字幕文件，保留换行等格式。
  • 💻 跨平台支持: 可在 Windows, macOS, 和 Linux 上运行。
  • 🚀 GPU 加速: 充分利用 NVIDIA GPU (CUDA) 进行运算，大幅提升处理速度。
## 🛠️ 环境配置与依赖安装
在运行脚本之前，请确保您的系统环境已正确配置。
### 1. 安装 FFmpeg
FFmpeg 是处理音视频的基础工具，必须预先安装并添加到系统的 PATH 环境变量中。
  • Windows:
    • 前往 。
    • 下载适用于 Windows 的编译版本。
    • 解压后，将 bin 文件夹的完整路径添加到系统的 Path 环境变量中。
  • macOS (使用 ):
```bash
brew install ffmpeg
```
  • Linux (以 Debian/Ubuntu 为例):
```bash
sudo apt update && sudo apt install ffmpeg
```
### 2. 安装 Python
请确保您的系统已安装 Python 3.8 或更高版本。
### 3. 安装 Python 依赖库
强烈建议在 Python 虚拟环境中安装依赖，以避免与系统库冲突。
  创建并激活虚拟环境 (推荐):
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```
  安装核心依赖:
    对于 NVIDIA GPU 用户 (强烈推荐): 为了获得最佳性能，请先根据您的 CUDA 版本从  安装支持 GPU 的 PyTorch，然后再安装其他包。
    通用安装命令:
```bash
pip install faster-whisper demucs pysubs2 "thefuzz[speedup]" torch torchaudio --upgrade
```
## 🚀 使用方法
通过命令行运行脚本，并指定必要的参数。
### 命令格式
```bash
python auto_timeline_script.py --video "你的视频文件.mp4" --subtitles "你的字幕文件.txt" [可选参数]
```
### 命令示例
假设你的视频是 my_lecture.mp4，无时间轴的字幕文件是 my_lecture_script.ass，你想使用 large-v3 模型并输出为 final_subs.ass。
```bash
python auto_timeline_script.py --video "D:\videos\my_lecture.mp4" --subtitles "D:\subs\my_lecture_script.ass" --model large-v3 --output "D:\output\final_subs.ass"
```
### 参数详解
参数 | 必填 | 默认值 | 描述
------------------
--video | 是 | 无 | 输入的视频文件路径。支持 
--subtitles | 是 | 无 | 不带时间轴的字幕文件路径。推荐使用 
--output | 否 | [视频文件名]_timed.ass | 指定输出的带时间轴字幕文件的路径和名称。
--model | 否 | medium | Whisper 模型大小。可选值：
--device | 否 | 自动检测 ( | 指定用于计算的设备。如果检测到可用的 NVIDIA GPU，则默认为 
--buffer | 否 | 0.05 | 为每行字幕的开始和结束时间添加的缓冲（单位：秒）。可以使字幕显示更自然，避免闪烁过快。
## ❓ 常见问题解答 (FAQ)
Q1: 处理速度非常慢怎么办？
> A: 速度主要受限于硬件和所选模型。
Q2: 识别或对齐的结果不准确怎么办？
> A: 准确性取决于多个因素。
Q3: 运行时出现 CUDA out of memory (CUDA 内存不足) 错误怎么办？
> A: 这个错误表示您的 GPU 显存不足以加载所选的模型。
Q4: 为什么有些字幕行被跳过了？
> A: 脚本在对齐时有一个置信度阈值。如果某行字幕在语音识别结果中找不到足够相似的匹配项（默认得分低于70），它将被跳过，并在控制台打印一条警告信息。这通常是因为该行字幕与实际语音内容差异太大。
## ⚠️ 注意事项
  • 输入文件编码: 为避免乱码，请确保您的输入字幕文件使用 UTF-8 编码。
  • 语言设置: 再次强调，脚本的默认语言是葡萄牙语。请务必根据您的视频内容，手动修改脚本中的 language 参数。
  • 临时文件: 脚本在运行时会在系统临时目录中创建中间文件（如提取的音频、分离的人声）。正常情况下，这些文件会在脚本结束时自动清理。如果脚本意外中断，可能需要手动删除这些文件。
  • 性能消耗: 人声分离 (Demucs) 和语音转录 (Whisper) 都是计算密集型任务，会消耗大量 CPU 或 GPU 资源，并可能需要较长时间，请耐心等待。
