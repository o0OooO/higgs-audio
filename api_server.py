#!/usr/bin/env python3
"""
FastAPI服务器，包装HiggsAudio音频生成功能
"""

import os
import base64
import io
import tempfile
import asyncio
import shutil
from typing import List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

# 导入HiggsAudio相关模块
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
from boson_multimodal.dataset.chatml_dataset import prepare_chatml_sample

# 配置日志
logger.add("api_server.log", rotation="10 MB", level="INFO")

# 创建FastAPI应用
app = FastAPI(
    title="HiggsAudio API",
    description="音频生成API，基于HiggsAudio模型",
    version="1.0.0"
)

# 全局变量存储模型实例
serve_engine: Optional[HiggsAudioServeEngine] = None


class AudioGenerationRequest(BaseModel):
    """音频生成请求模型"""
    text: str = Field(..., description="要转换为语音的文本")
    ref_audio: Optional[str] = Field(None, description="参考音频文件名（不包含扩展名）")
    scene_prompt: Optional[str] = Field("quiet_indoor", description="场景描述")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="生成温度")
    top_k: Optional[int] = Field(50, ge=1, le=100, description="Top-K采样参数")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Top-P采样参数")
    max_new_tokens: int = Field(2048, ge=1, le=4096, description="最大生成token数")
    seed: Optional[int] = Field(None, description="随机种子")
    ras_win_len: int = Field(7, ge=0, description="RAS窗口长度")
    ras_win_max_num_repeat: int = Field(2, ge=1, description="RAS最大重复次数")
    chunk_method: Optional[str] = Field(None, description="分块方法：speaker/word/None")
    chunk_max_word_num: int = Field(200, ge=1, description="单词分块最大词数")
    chunk_max_num_turns: int = Field(1, ge=1, description="说话人分块最大轮数")
    generation_chunk_buffer_size: Optional[int] = Field(None, description="生成分块缓冲区大小")


class AudioGenerationResponse(BaseModel):
    """音频生成响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    audio_base64: Optional[str] = Field(None, description="Base64编码的音频数据")
    sampling_rate: Optional[int] = Field(None, description="采样率")
    duration: Optional[float] = Field(None, description="音频时长（秒）")
    generated_text: Optional[str] = Field(None, description="生成的文本")


class ModelInfoResponse(BaseModel):
    """模型信息响应"""
    model_name: str
    device: str
    is_loaded: bool
    max_new_tokens: int
    supported_features: List[str]


@dataclass
class ServerConfig:
    """服务器配置"""
    model_path: str = "bosonai/higgs-audio-v2-generation-3B-base"
    audio_tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer"
    device: str = "auto"
    max_new_tokens: int = 2048
    voice_prompts_dir: str = "examples/voice_prompts"
    scene_prompts_dir: str = "examples/scene_prompts"


# 服务器配置
config = ServerConfig()


def get_device():
    """自动选择设备"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_model():
    """加载模型"""
    global serve_engine

    if serve_engine is not None:
        return serve_engine

    try:
        device = get_device() if config.device == "auto" else config.device
        logger.info(f"正在加载模型到设备: {device}")

        serve_engine = HiggsAudioServeEngine(
            model_name_or_path=config.model_path,
            audio_tokenizer_name_or_path=config.audio_tokenizer_path,
            device=device,
            torch_dtype="auto",
        )

        logger.info("模型加载成功")
        return serve_engine

    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    try:
        load_model()
        logger.info("API服务器启动成功")
    except Exception as e:
        logger.error(f"启动失败: {e}")


@app.get("/", response_model=dict)
async def root():
    """根路径"""
    return {
        "message": "HiggsAudio API服务器",
        "version": "1.0.0",
        "endpoints": [
            "/docs",
            "/info",
            "/generate",
            "/generate-stream",
            "/voices",
            "/scenes",
            "/upload-audio",
            "/upload-scene"
        ]
    }


@app.get("/info", response_model=ModelInfoResponse)
async def get_model_info():
    """获取模型信息"""
    try:
        engine = load_model()
        return ModelInfoResponse(
            model_name=config.model_path,
            device=get_device(),
            is_loaded=serve_engine is not None,
            max_new_tokens=config.max_new_tokens,
            supported_features=[
                "单说话人音频生成",
                "多说话人音频生成",
                "语音克隆",
                "智能语音",
                "场景描述",
                "流式生成"
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voices")
async def get_available_voices():
    """获取可用的语音样本"""
    try:
        voice_dir = Path(config.voice_prompts_dir)
        if not voice_dir.exists():
            return {"voices": [], "message": "语音样本目录不存在"}

        voices = []
        for file_path in voice_dir.glob("*.wav"):
            # 排除扩展名
            voice_name = file_path.stem
            voices.append({
                "name": voice_name,
                "file": str(file_path),
                "type": "wav"
            })

        return {"voices": voices, "count": len(voices)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scenes")
async def get_available_scenes():
    """获取可用的场景描述"""
    try:
        scene_dir = Path(config.scene_prompts_dir)
        if not scene_dir.exists():
            return {"scenes": [], "message": "场景描述目录不存在"}

        scenes = []
        for file_path in scene_dir.glob("*.txt"):
            scene_name = file_path.stem
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                scenes.append({
                    "name": scene_name,
                    "content": content,
                    "file": str(file_path)
                })
            except Exception as e:
                logger.warning(f"读取场景文件失败 {file_path}: {e}")

        return {"scenes": scenes, "count": len(scenes)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def prepare_messages(text: str, ref_audio: Optional[str] = None, scene_prompt: str = "quiet_indoor") -> ChatMLSample:
    """准备ChatML样本"""
    messages = []

    # 构建系统消息
    system_parts = ["Generate audio following instruction."]
    
    # 添加场景描述
    if scene_prompt and scene_prompt != "quiet_indoor":
        try:
            scene_file = Path(config.scene_prompts_dir) / f"{scene_prompt}.txt"
            if scene_file.exists():
                with open(scene_file, 'r', encoding='utf-8') as f:
                    scene_content = f.read().strip()
                system_parts.append(f"<|scene_desc_start|>\n{scene_content}\n<|scene_desc_end|>")
        except Exception as e:
            logger.warning(f"读取场景描述失败: {e}")
    
    # 添加参考音频描述
    if ref_audio:
        system_parts.append(f"Use reference audio '{ref_audio}' for voice characteristics.")
    
    system_content = "\n\n".join(system_parts)
    messages.append(Message(role="system", content=TextContent(text=system_content)))

    # 添加用户消息
    messages.append(Message(role="user", content=TextContent(text=text)))

    return ChatMLSample(messages=messages)


def audio_to_base64(audio_data: np.ndarray, sampling_rate: int) -> str:
    """将音频数据转换为Base64字符串"""
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, sampling_rate)

            # 读取文件并转换为base64
            with open(temp_file.name, 'rb') as f:
                audio_bytes = f.read()

            # 删除临时文件
            os.unlink(temp_file.name)

            return base64.b64encode(audio_bytes).decode('utf-8')

    except Exception as e:
        logger.error(f"音频转换失败: {e}")
        raise HTTPException(status_code=500, detail="音频转换失败")


@app.post("/generate", response_model=AudioGenerationResponse)
async def generate_audio(request: AudioGenerationRequest):
    """生成音频"""
    try:
        engine = load_model()

        # 准备ChatML样本
        chat_ml_sample = prepare_messages(
            text=request.text,
            ref_audio=request.ref_audio,
            scene_prompt=request.scene_prompt
        )

        # 生成音频
        response: HiggsAudioResponse = engine.generate(
            chat_ml_sample=chat_ml_sample,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            ras_win_len=request.ras_win_len,
            ras_win_max_num_repeat=request.ras_win_max_num_repeat,
            seed=request.seed,
        )

        if response.audio is None:
            raise HTTPException(status_code=500, detail="音频生成失败")

        # 转换为Base64
        audio_base64 = audio_to_base64(response.audio, response.sampling_rate)

        # 计算时长
        duration = len(response.audio) / response.sampling_rate

        return AudioGenerationResponse(
            success=True,
            message="音频生成成功",
            audio_base64=audio_base64,
            sampling_rate=response.sampling_rate,
            duration=duration,
            generated_text=response.generated_text
        )

    except Exception as e:
        logger.error(f"音频生成失败: {e}")
        return AudioGenerationResponse(
            success=False,
            message=f"音频生成失败: {str(e)}"
        )


@app.post("/generate-stream")
async def generate_audio_stream(request: AudioGenerationRequest):
    """流式生成音频"""
    try:
        engine = load_model()

        # 准备ChatML样本
        chat_ml_sample = prepare_messages(
            text=request.text,
            ref_audio=request.ref_audio,
            scene_prompt=request.scene_prompt
        )

        async def generate_stream():
            """生成流式响应"""
            try:
                async for delta in engine.generate_delta_stream(
                        chat_ml_sample=chat_ml_sample,
                        max_new_tokens=request.max_new_tokens,
                        temperature=request.temperature,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        ras_win_len=request.ras_win_len,
                        ras_win_max_num_repeat=request.ras_win_max_num_repeat,
                        seed=request.seed,
                ):
                    if delta.text:
                        yield f"data: {delta.text}\n\n"

                        # 发送音频token增量
                    if delta.audio_tokens is not None:
                        audio_tokens_base64 = base64.b64encode(
                            delta.audio_tokens.cpu().numpy().tobytes()
                        ).decode('utf-8')
                        yield f"data: AUDIO_TOKENS:{audio_tokens_base64}\n\n"

                    # 发送结束信号
                    yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"流式生成失败: {e}")
                yield f"data: ERROR:{str(e)}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )

    except Exception as e:
        logger.error(f"流式生成初始化失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-upload")
async def generate_audio_with_upload(
        text: str = Form(...),
        ref_audio_file: Optional[UploadFile] = File(None),
        temperature: float = Form(0.7),
        top_k: int = Form(50),
        top_p: float = Form(0.95),
        max_new_tokens: int = Form(2048),
        seed: Optional[int] = Form(None)
):
    """通过上传参考音频文件生成音频"""
    try:
        engine = load_model()

        # 处理上传的音频文件
        if ref_audio_file:
            # 读取上传的音频文件
            audio_content = await ref_audio_file.read()

            # 保存到临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_content)
                temp_file_path = temp_file.name

            # 这里需要将音频文件集成到消息中
            # 简化处理：暂时忽略上传的音频文件
            logger.warning("上传音频文件功能暂未完全实现")

        # 准备ChatML样本
        chat_ml_sample = prepare_messages(text=text)

        # 生成音频
        response: HiggsAudioResponse = engine.generate(
            chat_ml_sample=chat_ml_sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )

        if response.audio is None:
            raise HTTPException(status_code=500, detail="音频生成失败")

        # 返回音频文件
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, response.audio, response.sampling_rate, format='WAV')
        audio_bytes.seek(0)

        return StreamingResponse(
            io.BytesIO(audio_bytes.getvalue()),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=generated_audio.wav"}
        )

    except Exception as e:
        logger.error(f"音频生成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-audio")
async def upload_audio_file(audio_file: UploadFile = File(...)):
    """上传音频文件到语音克隆文件夹"""
    try:
        # 检查文件类型
        if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
            raise HTTPException(status_code=400, detail="只支持音频文件格式: wav, mp3, flac, m4a")
        
        # 确保目标目录存在
        voice_dir = Path(config.voice_prompts_dir)
        voice_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成安全的文件名
        filename = audio_file.filename
        # 移除扩展名
        name_without_ext = Path(filename).stem
        # 添加.wav扩展名（统一格式）
        safe_filename = f"{name_without_ext}.wav"
        file_path = voice_dir / safe_filename
        
        # 如果文件已存在，添加数字后缀
        counter = 1
        original_path = file_path
        while file_path.exists():
            safe_filename = f"{name_without_ext}_{counter}.wav"
            file_path = voice_dir / safe_filename
            counter += 1
        
        # 保存文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # 获取更新后的音频列表
        voices = []
        for file_path in voice_dir.glob("*.wav"):
            voice_name = file_path.stem
            voices.append({
                "name": voice_name,
                "file": str(file_path),
                "type": "wav"
            })
        
        return {
            "success": True,
            "message": f"音频文件上传成功: {safe_filename}",
            "uploaded_file": safe_filename,
            "voices": voices,
            "count": len(voices)
        }
        
    except Exception as e:
        logger.error(f"音频文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"音频文件上传失败: {str(e)}")


@app.post("/upload-scene")
async def upload_scene_file(scene_file: UploadFile = File(...)):
    """上传场景文本文件到场景文件夹"""
    try:
        # 检查文件类型
        if not scene_file.filename.lower().endswith('.txt'):
            raise HTTPException(status_code=400, detail="只支持文本文件格式: txt")
        
        # 确保目标目录存在
        scene_dir = Path(config.scene_prompts_dir)
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成安全的文件名
        filename = scene_file.filename
        # 移除扩展名
        name_without_ext = Path(filename).stem
        # 添加.txt扩展名
        safe_filename = f"{name_without_ext}.txt"
        file_path = scene_dir / safe_filename
        
        # 如果文件已存在，添加数字后缀
        counter = 1
        original_path = file_path
        while file_path.exists():
            safe_filename = f"{name_without_ext}_{counter}.txt"
            file_path = scene_dir / safe_filename
            counter += 1
        
        # 保存文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(scene_file.file, buffer)
        
        # 读取文件内容
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        except UnicodeDecodeError:
            # 如果UTF-8失败，尝试其他编码
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read().strip()
        
        # 获取更新后的场景列表
        scenes = []
        for file_path in scene_dir.glob("*.txt"):
            scene_name = file_path.stem
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    scene_content = f.read().strip()
                scenes.append({
                    "name": scene_name,
                    "content": scene_content,
                    "file": str(file_path)
                })
            except Exception as e:
                logger.warning(f"读取场景文件失败 {file_path}: {e}")
        
        return {
            "success": True,
            "message": f"场景文件上传成功: {safe_filename}",
            "uploaded_file": safe_filename,
            "content": content,
            "scenes": scenes,
            "count": len(scenes)
        }
        
    except Exception as e:
        logger.error(f"场景文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"场景文件上传失败: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HiggsAudio API服务器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8101, help="服务器端口")
    parser.add_argument("--model-path", default=config.model_path, help="模型路径")
    parser.add_argument("--audio-tokenizer-path", default=config.audio_tokenizer_path, help="音频tokenizer路径")
    parser.add_argument("--device", default=config.device, help="设备类型")
    parser.add_argument("--max-new-tokens", type=int, default=config.max_new_tokens, help="最大生成token数")
    parser.add_argument("--voice-prompts-dir", default=config.voice_prompts_dir, help="语音样本目录")
    parser.add_argument("--scene-prompts-dir", default=config.scene_prompts_dir, help="场景描述目录")
    parser.add_argument("--reload", action="store_true", help="开发模式重载")

    args = parser.parse_args()

    # 更新配置
    config.model_path = args.model_path
    config.audio_tokenizer_path = args.audio_tokenizer_path
    config.device = args.device
    config.max_new_tokens = args.max_new_tokens
    config.voice_prompts_dir = args.voice_prompts_dir
    config.scene_prompts_dir = args.scene_prompts_dir

    logger.info(f"启动HiggsAudio API服务器")
    logger.info(f"主机: {args.host}")
    logger.info(f"端口: {args.port}")
    logger.info(f"模型路径: {config.model_path}")
    logger.info(f"设备: {config.device}")

    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )