#!/usr/bin/env python3
"""
HiggsAudio API客户端测试脚本
"""

import requests
import json
import base64
import io
import soundfile as sf
from pathlib import Path


class HiggsAudioClient:
    """HiggsAudio API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_info(self):
        """获取服务器信息"""
        response = self.session.get(f"{self.base_url}/info")
        return response.json()
    
    def get_voices(self):
        """获取可用语音"""
        response = self.session.get(f"{self.base_url}/voices")
        return response.json()
    
    def get_scenes(self):
        """获取可用场景"""
        response = self.session.get(f"{self.base_url}/scenes")
        return response.json()
    
    def generate_audio(self, text: str, **kwargs):
        """生成音频"""
        data = {
            "text": text,
            **kwargs
        }
        response = self.session.post(f"{self.base_url}/generate", json=data)
        return response.json()
    
    def generate_audio_stream(self, text: str, **kwargs):
        """流式生成音频"""
        data = {
            "text": text,
            **kwargs
        }
        response = self.session.post(f"{self.base_url}/generate-stream", json=data, stream=True)
        return response
    
    def save_audio_from_base64(self, audio_base64: str, output_path: str):
        """从Base64保存音频文件"""
        try:
            # 解码Base64
            audio_bytes = base64.b64decode(audio_base64)
            
            # 保存为WAV文件
            with open(output_path, 'wb') as f:
                f.write(audio_bytes)
            
            print(f"音频已保存到: {output_path}")
            return True
        except Exception as e:
            print(f"保存音频失败: {e}")
            return False


def test_basic_functionality():
    """测试基本功能"""
    print("=== HiggsAudio API 客户端测试 ===\n")
    
    client = HiggsAudioClient()
    
    # 1. 获取服务器信息
    print("1. 获取服务器信息...")
    try:
        info = client.get_info()
        print(f"服务器信息: {json.dumps(info, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"获取服务器信息失败: {e}")
        return
    
    # 2. 获取可用语音
    print("\n2. 获取可用语音...")
    try:
        voices = client.get_voices()
        print(f"可用语音数量: {voices.get('count', 0)}")
        for voice in voices.get('voices', [])[:5]:  # 只显示前5个
            print(f"  - {voice['name']}")
    except Exception as e:
        print(f"获取语音列表失败: {e}")
    
    # 3. 获取可用场景
    print("\n3. 获取可用场景...")
    try:
        scenes = client.get_scenes()
        print(f"可用场景数量: {scenes.get('count', 0)}")
        for scene in scenes.get('scenes', [])[:3]:  # 只显示前3个
            print(f"  - {scene['name']}: {scene['content'][:50]}...")
    except Exception as e:
        print(f"获取场景列表失败: {e}")
    
    # 4. 测试音频生成
    print("\n4. 测试音频生成...")
    test_texts = [
        "Hello, this is a test of the HiggsAudio API.",
        "你好，这是HiggsAudio API的测试。",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n生成音频 {i+1}: {text}")
        try:
            result = client.generate_audio(
                text=text,
                temperature=0.7,
                max_new_tokens=512,
                seed=12345
            )
            
            if result.get('success'):
                print(f"  成功! 时长: {result.get('duration', 0):.2f}秒")
                print(f"  采样率: {result.get('sampling_rate', 0)}Hz")
                
                # 保存音频文件
                output_path = f"test_audio_{i+1}.wav"
                if client.save_audio_from_base64(result['audio_base64'], output_path):
                    print(f"  音频已保存: {output_path}")
            else:
                print(f"  失败: {result.get('message', '未知错误')}")
                
        except Exception as e:
            print(f"  生成失败: {e}")


def test_voice_cloning():
    """测试语音克隆功能"""
    print("\n=== 测试语音克隆功能 ===")
    
    client = HiggsAudioClient()
    
    # 获取可用语音
    voices = client.get_voices()
    available_voices = [v['name'] for v in voices.get('voices', [])]
    
    if not available_voices:
        print("没有可用的语音样本")
        return
    
    # 选择第一个语音进行测试
    test_voice = available_voices[0]
    test_text = "This is a voice cloning test using the HiggsAudio API."
    
    print(f"使用语音: {test_voice}")
    print(f"测试文本: {test_text}")
    
    try:
        result = client.generate_audio(
            text=test_text,
            ref_audio=test_voice,
            temperature=0.7,
            max_new_tokens=512,
            seed=12345
        )
        
        if result.get('success'):
            print("语音克隆成功!")
            output_path = f"voice_clone_{test_voice}.wav"
            if client.save_audio_from_base64(result['audio_base64'], output_path):
                print(f"音频已保存: {output_path}")
        else:
            print(f"语音克隆失败: {result.get('message', '未知错误')}")
            
    except Exception as e:
        print(f"语音克隆测试失败: {e}")


def test_streaming():
    """测试流式生成"""
    print("\n=== 测试流式生成 ===")
    
    client = HiggsAudioClient()
    test_text = "This is a streaming test of the HiggsAudio API."
    
    print(f"测试文本: {test_text}")
    
    try:
        response = client.generate_audio_stream(
            text=test_text,
            temperature=0.7,
            max_new_tokens=512,
            seed=12345
        )
        
        print("流式响应:")
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data = line_str[6:]  # 移除 'data: ' 前缀
                    if data == '[DONE]':
                        print("  生成完成")
                        break
                    elif data.startswith('ERROR:'):
                        print(f"  错误: {data[6:]}")
                        break
                    elif data.startswith('AUDIO_TOKENS:'):
                        print("  收到音频token")
                    else:
                        print(f"  文本: {data}")
                        
    except Exception as e:
        print(f"流式生成测试失败: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HiggsAudio API客户端测试")
    parser.add_argument("--url", default="http://localhost:8000", help="API服务器URL")
    parser.add_argument("--test", choices=["basic", "voice-clone", "streaming", "all"], 
                       default="all", help="测试类型")
    
    args = parser.parse_args()
    
    # 更新客户端URL
    client = HiggsAudioClient(args.url)
    
    if args.test == "basic" or args.test == "all":
        test_basic_functionality()
    
    if args.test == "voice-clone" or args.test == "all":
        test_voice_cloning()
    
    if args.test == "streaming" or args.test == "all":
        test_streaming()
    
    print("\n=== 测试完成 ===") 