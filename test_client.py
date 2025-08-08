#!/usr/bin/env python3
"""
HiggsAudio API客户端测试脚本
"""

import requests
import json
import base64
import os


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
    
    def upload_audio_file(self, file_path: str):
        """上传音频文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        with open(file_path, 'rb') as f:
            files = {'audio_file': (os.path.basename(file_path), f, 'audio/wav')}
            response = self.session.post(f"{self.base_url}/upload-audio", files=files)
        
        return response.json()
    
    def upload_scene_file(self, file_path: str):
        """上传场景文本文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        with open(file_path, 'rb') as f:
            files = {'scene_file': (os.path.basename(file_path), f, 'text/plain')}
            response = self.session.post(f"{self.base_url}/upload-scene", files=files)
        
        return response.json()
    
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


def test_upload_audio():
    """测试上传音频文件功能"""
    print("\n=== 测试上传音频文件功能 ===")
    
    client = HiggsAudioClient()
    
    # 创建一个测试音频文件（如果不存在）
    test_audio_path = "test_upload_audio.wav"
    if not os.path.exists(test_audio_path):
        print(f"创建测试音频文件: {test_audio_path}")
        # 创建一个简单的测试音频文件（这里只是示例，实际应该是一个真实的音频文件）
        import wave
        import struct
        
        # 创建一个简单的正弦波音频文件
        sample_rate = 44100
        duration = 1  # 1秒
        frequency = 440  # 440 Hz
        
        with wave.open(test_audio_path, 'w') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位
            wav_file.setframerate(sample_rate)
            
            for i in range(sample_rate * duration):
                # 生成正弦波
                value = int(32767 * 0.3 * (i * frequency * 2 * 3.14159 / sample_rate))
                # 确保值在16位整数范围内
                value = max(-32768, min(32767, value))
                data = struct.pack('<h', value)
                wav_file.writeframes(data)
    
    print(f"上传音频文件: {test_audio_path}")
    
    try:
        result = client.upload_audio_file(test_audio_path)
        
        if result.get('success'):
            print("音频文件上传成功!")
            print(f"上传的文件: {result.get('uploaded_file')}")
            print(f"当前音频总数: {result.get('count', 0)}")
            
            # 显示部分音频列表
            voices = result.get('voices', [])
            print("音频列表:")
            for voice in voices[-5:]:  # 显示最后5个
                print(f"  - {voice['name']}")
        else:
            print(f"音频文件上传失败: {result.get('message', '未知错误')}")
            
    except Exception as e:
        print(f"音频文件上传测试失败: {e}")


def test_upload_scene():
    """测试上传场景文本文件功能"""
    print("\n=== 测试上传场景文本文件功能 ===")
    
    client = HiggsAudioClient()
    
    # 创建一个测试场景文件
    test_scene_path = "test_upload_scene.txt"
    test_scene_content = """这是一个测试场景描述。
场景包含：安静的室内环境，有轻微的空调声，背景有轻柔的音乐。
适合：语音录制、播客制作、有声书朗读等场景。"""
    
    print(f"创建测试场景文件: {test_scene_path}")
    with open(test_scene_path, 'w', encoding='utf-8') as f:
        f.write(test_scene_content)
    
    print(f"上传场景文件: {test_scene_path}")
    
    try:
        result = client.upload_scene_file(test_scene_path)
        
        if result.get('success'):
            print("场景文件上传成功!")
            print(f"上传的文件: {result.get('uploaded_file')}")
            print(f"文件内容: {result.get('content', '')[:100]}...")
            print(f"当前场景总数: {result.get('count', 0)}")
            
            # 显示部分场景列表
            scenes = result.get('scenes', [])
            print("场景列表:")
            for scene in scenes[-5:]:  # 显示最后5个
                print(f"  - {scene['name']}: {scene['content'][:50]}...")
        else:
            print(f"场景文件上传失败: {result.get('message', '未知错误')}")
            
    except Exception as e:
        print(f"场景文件上传测试失败: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HiggsAudio API客户端测试")
    parser.add_argument("--url", default="http://localhost:8000", help="API服务器URL")
    parser.add_argument("--test", choices=["basic", "voice-clone", "streaming", "upload-audio", "upload-scene", "all"], 
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
    
    if args.test == "upload-audio" or args.test == "all":
        test_upload_audio()
    
    if args.test == "upload-scene" or args.test == "all":
        test_upload_scene()
    
    print("\n=== 测试完成 ===") 