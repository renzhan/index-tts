#!/usr/bin/env python3
"""
IndexTTS2 FastAPI API服务器
兼容OpenAI TTS接口，支持通过参考音频路径进行语音合成
"""

import os
import sys
import tempfile
import uuid
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import torch
import torchaudio
import io
import base64

# 导入IndexTTS2
from indextts.infer_v2 import IndexTTS2

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
tts_model = None
model_dir = "models/IndexTTS-2"
config_path = os.path.join(model_dir, "config.yaml")

# 创建FastAPI应用
app = FastAPI(
    title="IndexTTS2 API Server",
    description="IndexTTS2语音合成API服务，兼容OpenAI TTS接口",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class TTSRequest(BaseModel):
    """TTS请求模型"""
    text: str = Field(..., description="要合成的文本")
    voice: str = Field(..., description="参考音频文件路径")
    model: str = Field(default="indextts2", description="模型名称")
    response_format: str = Field(default="wav", description="响应格式 (wav, mp3)")
    speed: float = Field(default=1.0, description="语速倍数")
    emo_audio_prompt: Optional[str] = Field(None, description="情感参考音频路径")
    emo_alpha: float = Field(default=1.0, description="情感强度 (0.0-1.0)")
    emo_vector: Optional[List[float]] = Field(None, description="情感向量 [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]")
    use_emo_text: bool = Field(default=False, description="是否使用文本情感分析")
    emo_text: Optional[str] = Field(None, description="情感描述文本")
    use_random: bool = Field(default=False, description="是否使用随机采样")
    verbose: bool = Field(default=False, description="是否输出详细信息")

class OpenAICompatibleRequest(BaseModel):
    """OpenAI兼容的TTS请求模型"""
    text: str = Field(..., description="要合成的文本")
    voice: str = Field(..., description="参考音频文件路径")
    model: str = Field(default="indextts2", description="模型名称")
    response_format: str = Field(default="wav", description="响应格式")
    speed: float = Field(default=1.0, description="语速倍数")

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    device: str
    timestamp: float

# 初始化模型
def initialize_model():
    """初始化IndexTTS2模型"""
    global tts_model
    
    try:
        logger.info("正在初始化IndexTTS2模型...")
        
        # 检查模型文件是否存在
        required_files = [
            "config.yaml",
            "gpt.pth", 
            "s2mel.pth",
            "bpe.model",
            "wav2vec2bert_stats.pt",
            "feat1.pt",
            "feat2.pt"
        ]
        
        for file in required_files:
            file_path = os.path.join(model_dir, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"缺少必要文件: {file_path}")
        
        # 初始化模型
        tts_model = IndexTTS2(
            cfg_path=config_path,
            model_dir=model_dir,
            use_fp16=True,
            use_cuda_kernel=torch.cuda.is_available(),
            use_deepspeed=False,
            device=None
        )
        
        logger.info("IndexTTS2模型初始化完成！")
        return True
        
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        return False

# 启动时初始化模型
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模型"""
    success = initialize_model()
    if not success:
        logger.error("模型初始化失败，API服务可能无法正常工作")

# API端点
@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径"""
    return {
        "message": "IndexTTS2 API Server",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy" if tts_model is not None else "unhealthy",
        model_loaded=tts_model is not None,
        device=str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
        timestamp=time.time()
    )

@app.post("/v1/audio/speech")
async def create_speech_openai_compatible(request: OpenAICompatibleRequest):
    """
    OpenAI兼容的TTS接口
    兼容OpenAI的 /v1/audio/speech 端点
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 检查参考音频文件是否存在
        if not os.path.exists(request.voice):
            raise HTTPException(status_code=400, detail=f"参考音频文件不存在: {request.voice}")
        
        # 生成唯一文件名
        output_filename = f"speech_{uuid.uuid4().hex}.wav"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        
        # 进行语音合成
        logger.info(f"开始合成语音: {request.text[:50]}...")
        
        result_path = tts_model.infer(
            spk_audio_prompt=request.voice,
            text=request.text,
            output_path=output_path,
            verbose=request.speed != 1.0  # 如果语速不是1.0则输出详细信息
        )
        
        # 返回音频文件
        return FileResponse(
            path=result_path,
            media_type="audio/wav",
            filename=f"speech_{int(time.time())}.wav"
        )
        
    except Exception as e:
        logger.error(f"语音合成失败: {e}")
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")

@app.post("/api/v1/tts")
async def create_speech(request: TTSRequest):
    """
    IndexTTS2完整功能TTS接口
    支持所有IndexTTS2的高级功能
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 检查参考音频文件是否存在
        if not os.path.exists(request.voice):
            raise HTTPException(status_code=400, detail=f"参考音频文件不存在: {request.voice}")
        
        # 检查情感参考音频文件（如果提供）
        if request.emo_audio_prompt and not os.path.exists(request.emo_audio_prompt):
            raise HTTPException(status_code=400, detail=f"情感参考音频文件不存在: {request.emo_audio_prompt}")
        
        # 生成唯一文件名
        output_filename = f"speech_{uuid.uuid4().hex}.wav"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        
        # 进行语音合成
        logger.info(f"开始合成语音: {request.text[:50]}...")
        
        result_path = tts_model.infer(
            spk_audio_prompt=request.voice,
            text=request.text,
            output_path=output_path,
            emo_audio_prompt=request.emo_audio_prompt,
            emo_alpha=request.emo_alpha,
            emo_vector=request.emo_vector,
            use_emo_text=request.use_emo_text,
            emo_text=request.emo_text,
            use_random=request.use_random,
            verbose=request.verbose
        )
        
        # 返回音频文件
        return FileResponse(
            path=result_path,
            media_type="audio/wav",
            filename=f"speech_{int(time.time())}.wav"
        )
        
    except Exception as e:
        logger.error(f"语音合成失败: {e}")
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")

@app.post("/api/v1/tts/upload")
async def create_speech_with_upload(
    text: str = Form(...),
    voice_file: UploadFile = File(...),
    emo_audio_file: Optional[UploadFile] = File(None),
    emo_alpha: float = Form(1.0),
    emo_vector: Optional[str] = Form(None),
    use_emo_text: bool = Form(False),
    emo_text: Optional[str] = Form(None),
    use_random: bool = Form(False),
    verbose: bool = Form(False)
):
    """
    支持文件上传的TTS接口
    可以直接上传参考音频文件
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 保存上传的参考音频文件
        voice_filename = f"voice_{uuid.uuid4().hex}.wav"
        voice_path = os.path.join(tempfile.gettempdir(), voice_filename)
        
        with open(voice_path, "wb") as f:
            content = await voice_file.read()
            f.write(content)
        
        # 保存情感参考音频文件（如果提供）
        emo_audio_path = None
        if emo_audio_file:
            emo_filename = f"emo_{uuid.uuid4().hex}.wav"
            emo_audio_path = os.path.join(tempfile.gettempdir(), emo_filename)
            
            with open(emo_audio_path, "wb") as f:
                content = await emo_audio_file.read()
                f.write(content)
        
        # 解析情感向量
        emo_vector_list = None
        if emo_vector:
            try:
                emo_vector_list = [float(x.strip()) for x in emo_vector.split(",")]
                if len(emo_vector_list) != 8:
                    raise ValueError("情感向量必须包含8个值")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"情感向量格式错误: {e}")
        
        # 生成输出文件名
        output_filename = f"speech_{uuid.uuid4().hex}.wav"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        
        # 进行语音合成
        logger.info(f"开始合成语音: {text[:50]}...")
        
        result_path = tts_model.infer(
            spk_audio_prompt=voice_path,
            text=text,
            output_path=output_path,
            emo_audio_prompt=emo_audio_path,
            emo_alpha=emo_alpha,
            emo_vector=emo_vector_list,
            use_emo_text=use_emo_text,
            emo_text=emo_text,
            use_random=use_random,
            verbose=verbose
        )
        
        # 清理临时文件
        try:
            os.remove(voice_path)
            if emo_audio_path:
                os.remove(emo_audio_path)
        except:
            pass
        
        # 返回音频文件
        return FileResponse(
            path=result_path,
            media_type="audio/wav",
            filename=f"speech_{int(time.time())}.wav"
        )
        
    except Exception as e:
        logger.error(f"语音合成失败: {e}")
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")

@app.get("/api/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "indextts2",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "IndexTTS2",
                "permission": [],
                "root": "indextts2",
                "parent": None
            }
        ]
    }

@app.get("/api/v1/voices")
async def list_voices():
    """列出可用声音（这里返回示例）"""
    return {
        "object": "list",
        "data": [
            {
                "id": "custom",
                "object": "voice",
                "name": "Custom Voice",
                "description": "使用参考音频文件自定义声音"
            }
        ]
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IndexTTS2 API Server")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--model-dir", default="models/IndexTTS-2", help="模型目录")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--reload", action="store_true", help="开发模式，自动重载")
    
    args = parser.parse_args()
    
    # 更新全局变量
    model_dir = args.model_dir
    config_path = os.path.join(model_dir, "config.yaml")
    
    # 启动服务器
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )
