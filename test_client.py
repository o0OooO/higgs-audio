#!/usr/bin/env python3
"""
HiggsAudio API客户端测试脚本
"""

import requests
import json
import base64
import os
from pathlib import Path


class HiggsAudioClient:
    """HiggsAudio API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8101"):
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
    
    def get_profiles(self):
        """获取所有Profile"""
        response = self.session.get(f"{self.base_url}/profiles")
        return response.json()
    
    def get_profile(self, profile_name: str):
        """获取指定Profile"""
        response = self.session.get(f"{self.base_url}/profiles/{profile_name}")
        return response.json()
    
    def create_profile(self, name: str, description: str, characteristics: dict = None):
        """创建新Profile"""
        data = {
            "name": name,
            "description": description,
            "characteristics": characteristics or {}
        }
        response = self.session.post(f"{self.base_url}/profiles", json=data)
        return response.json()
    
    def update_profile(self, profile_name: str, description: str = None, characteristics: dict = None):
        """更新Profile"""
        data = {}
        if description is not None:
            data["description"] = description
        if characteristics is not None:
            data["characteristics"] = characteristics
        
        response = self.session.put(f"{self.base_url}/profiles/{profile_name}", json=data)
        return response.json()
    
    def delete_profile(self, profile_name: str):
        """删除Profile"""
        response = self.session.delete(f"{self.base_url}/profiles/{profile_name}")
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


def _ensure_output_dir(dir_path: str) -> str:
    """确保输出目录存在并返回目录路径"""
    out_dir = Path(dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)


def _read_text_file(file_path: Path) -> str:
    """读取文本/Markdown文件内容（utf-8优先）"""
    try:
        return file_path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return file_path.read_text(encoding="gbk").strip()


def _run_case_and_save(client: HiggsAudioClient, text: str, output_path: Path, **kwargs) -> bool:
    """通用用例执行与落盘"""
    try:
        result = client.generate_audio(text=text, **kwargs)
        if result.get('success'):
            print(f"  成功! 时长: {result.get('duration', 0):.2f}秒 | 采样率: {result.get('sampling_rate', 0)}Hz")
            ok = client.save_audio_from_base64(result['audio_base64'], str(output_path))
            if ok:
                print(f"  音频已保存: {output_path}")
            return ok
        else:
            print(f"  失败: {result.get('message', '未知错误')}")
            return False
    except Exception as e:
        print(f"  生成失败: {e}")
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


def test_smart_voice_single_speaker(language: str = "en", output_dir: str = "outputs"):
    """测试智能随机音色的单人朗读（英文/中文）"""
    print("\n=== 测试智能音色（单说话人） ===")
    client = HiggsAudioClient()
    base = Path(__file__).parent
    transcript_map = {
        "en": base / "examples/transcript/single_speaker/en_dl.txt",
        "zh": base / "examples/transcript/single_speaker/zh_ai.txt",
    }
    transcript_path = transcript_map.get(language, transcript_map["en"]) 
    text = _read_text_file(transcript_path)
    _ensure_output_dir(output_dir)
    out_path = Path(output_dir) / f"smart_voice_single_{language}.wav"
    print(f"使用文稿: {transcript_path}")
    _run_case_and_save(
        client,
        text=text,
        output_path=out_path,
        seed=12345,
        temperature=0.7,
        max_new_tokens=2048,
    )


def test_voice_profile_british(output_dir: str = "outputs"):
    """测试基于文本画像（英式口音：男/女）"""
    print("\n=== 测试文本语音画像（英式口音） ===")
    client = HiggsAudioClient()
    base = Path(__file__).parent
    transcript_path = base / "examples/transcript/single_speaker/en_dl.txt"
    text = _read_text_file(transcript_path)
    _ensure_output_dir(output_dir)

    cases = [
        ("profile_male_en_british.wav", "profile:male_en_british"),
        ("profile_female_en_british.wav", "profile:female_en_british"),
    ]
    print(f"使用文稿: {transcript_path}")
    for filename, ref in cases:
        print(f"\n画像: {ref}")
        _run_case_and_save(
            client,
            text=text,
            output_path=Path(output_dir) / filename,
            ref_audio=ref,
            seed=12345,
            temperature=0.7,
            max_new_tokens=2048,
        )


def test_cross_lingual_voice_clone(output_dir: str = "outputs"):
    """测试跨语种克隆（中文音色，英文朗读）"""
    print("\n=== 测试跨语种语音克隆 ===")
    client = HiggsAudioClient()
    base = Path(__file__).parent
    transcript_path = base / "examples/transcript/single_speaker/en_dl.txt"
    text = _read_text_file(transcript_path)
    _ensure_output_dir(output_dir)
    print(f"使用文稿: {transcript_path}")
    _run_case_and_save(
        client,
        text=text,
        output_path=Path(output_dir) / "cross_lingual_zh_sichuan_to_en.wav",
        ref_audio="zh_man_sichuan",
        scene_prompt="empty",  # 若文件不存在则等效不加场景
        temperature=0.3,
        seed=12345,
        max_new_tokens=2048,
    )


def test_humming_with_cloned_voice(output_dir: str = "outputs"):
    """测试哼唱能力（使用英文女声画像）"""
    print("\n=== 测试哼唱（Humming） ===")
    client = HiggsAudioClient()
    base = Path(__file__).parent
    transcript_path = base / "examples/transcript/single_speaker/experimental/en_humming.txt"
    text = _read_text_file(transcript_path)
    _ensure_output_dir(output_dir)
    print(f"使用文稿: {transcript_path}")
    _run_case_and_save(
        client,
        text=text,
        output_path=Path(output_dir) / "humming_en_woman.wav",
        ref_audio="en_woman",
        ras_win_len=0,
        seed=12345,
        max_new_tokens=2048,
    )


def test_bgm_reading(output_dir: str = "outputs"):
    """测试带背景音乐（BGM）的朗读"""
    print("\n=== 测试背景音乐（BGM） ===")
    client = HiggsAudioClient()
    base = Path(__file__).parent
    transcript_path = base / "examples/transcript/single_speaker/experimental/en_bgm.txt"
    text = _read_text_file(transcript_path)
    _ensure_output_dir(output_dir)
    print(f"使用文稿: {transcript_path}")
    _run_case_and_save(
        client,
        text=text,
        output_path=Path(output_dir) / "bgm_reading_en_woman.wav",
        ref_audio="en_woman",
        ras_win_len=0,
        seed=123456,
        max_new_tokens=2048,
    )


def test_multi_speaker_zero_shot(output_dir: str = "outputs"):
    """测试多人对话（零样本，自动分配音色）"""
    print("\n=== 测试多人对话（零样本） ===")
    client = HiggsAudioClient()
    base = Path(__file__).parent
    transcript_path = base / "examples/transcript/multi_speaker/en_argument.txt"
    text = _read_text_file(transcript_path)
    _ensure_output_dir(output_dir)
    print(f"使用文稿: {transcript_path}")
    _run_case_and_save(
        client,
        text=text,
        output_path=Path(output_dir) / "multi_speaker_zero_shot.wav",
        seed=12345,
        max_new_tokens=2048,
        chunk_method="speaker",  # 服务器当前可能忽略该参数，但保留以对齐示例
    )


# def test_multi_voice_clone_argument(output_dir: str = "outputs"):
#     """测试多人语音克隆（Belinda vs Broom Salesman 争论）"""
#     print("\n=== 测试多人语音克隆：Belinda 与 Broom Salesman ===")
#     client = HiggsAudioClient()
#     base = Path(__file__).parent
#     transcript_path = base / "examples/transcript/multi_speaker/en_argument.txt"
#     text = _read_text_file(transcript_path)
#     _ensure_output_dir(output_dir)
#     print(f"使用文稿: {transcript_path}")
#     _run_case_and_save(
#         client,
#         text=text,
#         output_path=Path(output_dir) / "multi_clone_belinda_broomsalesman.wav",
#         ref_audio="belinda,broom_salesman",
#         seed=12345,
#         max_new_tokens=2048,
#         chunk_method="speaker",
#     )


def test_multi_voice_clone_argument(output_dir: str = "outputs"):
    """测试多人语音克隆（Belinda vs Broom Salesman 争论）"""
    print("\n=== 测试多人语音克隆：Belinda 与 Broom Salesman ===")
    client = HiggsAudioClient()
    base = Path(__file__).parent
    transcript_path = base / "examples/transcript/multi_speaker/en_argument.txt"
    text = _read_text_file(transcript_path)
    _ensure_output_dir(output_dir)
    print(f"使用文稿: {transcript_path}")
    _run_case_and_save(
        client,
        text=text,
        output_path=Path(output_dir) / "multi_clone_belinda_broomsalesman.wav",
        ref_audio="profile:male_en,profile:female_en_story",
        seed=12345,
        max_new_tokens=2048,
        chunk_method="speaker",
    )



def test_multi_voice_clone_higgs_dialog(output_dir: str = "outputs"):
    """测试多人语音克隆（Broom Salesman 与 Belinda 讨论 HiggsAudio）"""
    print("\n=== 测试多人语音克隆：Broom Salesman 与 Belinda（Higgs 话题） ===")
    client = HiggsAudioClient()
    base = Path(__file__).parent
    transcript_path = base / "examples/transcript/multi_speaker/en_higgs.txt"
    text = _read_text_file(transcript_path)
    _ensure_output_dir(output_dir)
    print(f"使用文稿: {transcript_path}")
    _run_case_and_save(
        client,
        text=text,
        output_path=Path(output_dir) / "multi_clone_higgs_dialog.wav",
        ref_audio="broom_salesman,belinda",
        seed=12345,
        max_new_tokens=2048,
        chunk_method="speaker",
        chunk_max_num_turns=2,
    )


def test_emotion_styles(output_dir: str = "outputs"):
    """测试情绪风格（愉快/愤怒/悲伤）"""
    print("\n=== 测试情绪风格（happy / angry / sad） ===")
    client = HiggsAudioClient()
    _ensure_output_dir(output_dir)
    prompts = [
        ("emotion_happy.wav", "Please read the sentence in a happy and excited tone: Today is a wonderful day!"),
        ("emotion_angry.wav", "Please read the sentence in an angry tone with strong emphasis: Why didn't you tell me earlier?"),
        ("emotion_sad.wav", "Please read the sentence in a calm and sad tone: I really miss those times.")
    ]
    for filename, text in prompts:
        print(f"\n情绪用例: {filename}")
        _run_case_and_save(
            client,
            text=text,
            output_path=Path(output_dir) / filename,
            seed=12345,
            temperature=0.7,
            max_new_tokens=2048,
        )


def test_long_form_blog_reading(output_dir: str = "outputs"):
    """测试长文分块朗读（博客前几段）"""
    print("\n=== 测试长文阅读（博客节选） ===")
    client = HiggsAudioClient()
    base = Path(__file__).parent
    transcript_path = base / "examples/transcript/single_speaker/en_higgs_audio_blog.md"
    text = _read_text_file(transcript_path)
    _ensure_output_dir(output_dir)
    print(f"使用文稿: {transcript_path}")
    _run_case_and_save(
        client,
        text=text,
        output_path=Path(output_dir) / "long_form_blog_reading.wav",
        ref_audio="en_man",
        scene_prompt="reading_blog",
        temperature=0.3,
        seed=12345,
        max_new_tokens=2048,
        chunk_method="word",  # 当前服务端可能未启用分块，这里保持参数对齐
        generation_chunk_buffer_size=2,
    )


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


def test_profile_management():
    """测试Profile管理功能"""
    print("\n=== 测试Profile管理功能 ===")
    
    client = HiggsAudioClient()
    
    # 1. 获取所有Profile
    print("1. 获取所有Profile...")
    try:
        result = client.get_profiles()
        if result.get('success'):
            profiles = result.get('profiles', [])
            print(f"当前Profile数量: {result.get('count', 0)}")
            for profile in profiles:
                print(f"  - {profile['name']}: {profile['description'][:50]}...")
        else:
            print(f"获取Profile列表失败: {result.get('message', '未知错误')}")
    except Exception as e:
        print(f"获取Profile列表失败: {e}")
    
    # 2. 获取特定Profile
    print("\n2. 获取特定Profile...")
    try:
        result = client.get_profile("male_en_british")
        if result.get('success'):
            profile = result.get('profile')
            print(f"Profile: {profile['name']}")
            print(f"描述: {profile['description']}")
        else:
            print(f"获取Profile失败: {result.get('message', '未知错误')}")
    except Exception as e:
        print(f"获取Profile失败: {e}")
    
    # 3. 创建新Profile
    print("\n3. 创建新Profile...")
    try:
        new_profile_name = "test_profile"
        new_description = "这是一个测试Profile，用于验证API功能。"
        characteristics = {
            "accent": "American",
            "gender": "male",
            "tone": "professional",
            "speed": "moderate"
        }
        
        result = client.create_profile(
            name=new_profile_name,
            description=new_description,
            characteristics=characteristics
        )
        
        if result.get('success'):
            print("Profile创建成功!")
            profile = result.get('profile')
            print(f"名称: {profile['name']}")
            print(f"描述: {profile['description']}")
        else:
            print(f"Profile创建失败: {result.get('message', '未知错误')}")
    except Exception as e:
        print(f"Profile创建失败: {e}")
    
    # 4. 更新Profile
    print("\n4. 更新Profile...")
    try:
        updated_description = "这是更新后的测试Profile描述。"
        updated_characteristics = {
            "accent": "American",
            "gender": "male",
            "tone": "friendly",
            "speed": "fast"
        }
        
        result = client.update_profile(
            profile_name=new_profile_name,
            description=updated_description,
            characteristics=updated_characteristics
        )
        
        if result.get('success'):
            print("Profile更新成功!")
            profile = result.get('profile')
            print(f"名称: {profile['name']}")
            print(f"描述: {profile['description']}")
        else:
            print(f"Profile更新失败: {result.get('message', '未知错误')}")
    except Exception as e:
        print(f"Profile更新失败: {e}")
    
    # 5. 验证更新后的Profile
    print("\n5. 验证更新后的Profile...")
    try:
        result = client.get_profile(new_profile_name)
        if result.get('success'):
            profile = result.get('profile')
            print(f"验证成功! 更新后的描述: {profile['description']}")
        else:
            print(f"验证Profile失败: {result.get('message', '未知错误')}")
    except Exception as e:
        print(f"验证Profile失败: {e}")
    
    # 6. 删除测试Profile
    # print("\n6. 删除测试Profile...")
    # try:
    #     result = client.delete_profile(new_profile_name)
    #     if result.get('success'):
    #         print("Profile删除成功!")
    #     else:
    #         print(f"Profile删除失败: {result.get('message', '未知错误')}")
    # except Exception as e:
    #     print(f"Profile删除失败: {e}")
    
    # 7. 验证删除结果
    print("\n7. 验证删除结果...")
    try:
        result = client.get_profile(new_profile_name)
        if not result.get('success'):
            print("Profile删除验证成功!")
        else:
            print("Profile删除验证失败，Profile仍然存在")
    except Exception as e:
        print(f"Profile删除验证失败: {e}")


def test_profile_with_audio_generation():
    """测试使用Profile进行音频生成"""
    print("\n=== 测试使用Profile进行音频生成 ===")
    
    client = HiggsAudioClient()
    
    # 获取可用的Profile
    try:
        result = client.get_profiles()
        if not result.get('success'):
            print("无法获取Profile列表，跳过测试")
            return
        
        profiles = result.get('profiles', [])
        if not profiles:
            print("没有可用的Profile，跳过测试")
            return
        
        # 选择第一个Profile进行测试
        test_profile = profiles[0]
        test_text = "This is a test of audio generation using a voice profile."
        
        print(f"使用Profile: {test_profile['name']}")
        print(f"Profile描述: {test_profile['description']}")
        print(f"测试文本: {test_text}")
        
        # 生成音频
        result = client.generate_audio(
            text=test_text,
            ref_audio=f"profile:{test_profile['name']}",
            temperature=0.7,
            max_new_tokens=512,
            seed=12345
        )
        
        if result.get('success'):
            print("使用Profile生成音频成功!")
            output_path = f"profile_audio_{test_profile['name']}.wav"
            if client.save_audio_from_base64(result['audio_base64'], output_path):
                print(f"音频已保存: {output_path}")
        else:
            print(f"使用Profile生成音频失败: {result.get('message', '未知错误')}")
            
    except Exception as e:
        print(f"Profile音频生成测试失败: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HiggsAudio API客户端测试")
    parser.add_argument("--url", default="http://localhost:8101", help="API服务器URL")
    parser.add_argument("--output-dir", default="outputs", help="生成音频输出目录")
    parser.add_argument(
        "--test",
        choices=[
            "basic",
            "voice-clone",
            "streaming",
            "upload-audio",
            "upload-scene",
            # 新增场景化测试
            "smart-voice-en",
            "smart-voice-zh",
            "profile-voices",
            "cross-lingual",
            "humming",
            "bgm",
            "multi-speaker-zero-shot",
            "multi-voice-clone",
            "multi-voice-clone-higgs",
            "emotions",
            "long-form-blog",
            # Profile管理测试
            "profile-management",
            "profile-with-audio-generation",
            # 组合项
            "all",
            "all-scenarios",
        ],
        default="all",
        help="测试类型",
    )
    
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

    # 单场景执行
    if args.test == "smart-voice-en" or args.test == "all":
        test_smart_voice_single_speaker("en", args.output_dir)
    if args.test == "smart-voice-zh" or args.test == "all":
        test_smart_voice_single_speaker("zh", args.output_dir)
    if args.test == "profile-voices" or args.test == "all":
        test_voice_profile_british(args.output_dir)
    if args.test == "cross-lingual" or args.test == "all":
        test_cross_lingual_voice_clone(args.output_dir)
    if args.test == "humming" or args.test == "all":
        test_humming_with_cloned_voice(args.output_dir)
    if args.test == "bgm" or args.test == "all":
        test_bgm_reading(args.output_dir)
    if args.test == "multi-speaker-zero-shot" or args.test == "all":
        test_multi_speaker_zero_shot(args.output_dir)
    if args.test == "multi-voice-clone" or args.test == "all":
        test_multi_voice_clone_argument(args.output_dir)
    if args.test == "multi-voice-clone-higgs" or args.test == "all":
        test_multi_voice_clone_higgs_dialog(args.output_dir)
    if args.test == "emotions" or args.test == "all":
        test_emotion_styles(args.output_dir)
    if args.test == "long-form-blog" or args.test == "all":
        test_long_form_blog_reading(args.output_dir)
    if args.test == "profile-management" or args.test == "all":
        test_profile_management()
    if args.test == "profile-with-audio-generation" or args.test == "all":
        test_profile_with_audio_generation()

    # 运行所有扩展示例
    if args.test == "all-scenarios":
        test_smart_voice_single_speaker("en", args.output_dir)
        test_smart_voice_single_speaker("zh", args.output_dir)
        test_voice_profile_british(args.output_dir)
        test_cross_lingual_voice_clone(args.output_dir)
        test_humming_with_cloned_voice(args.output_dir)
        test_bgm_reading(args.output_dir)
        test_multi_speaker_zero_shot(args.output_dir)
        test_multi_voice_clone_argument(args.output_dir)
        test_multi_voice_clone_higgs_dialog(args.output_dir)
        test_emotion_styles(args.output_dir)
        test_long_form_blog_reading(args.output_dir)
        test_profile_management()
        test_profile_with_audio_generation()
    
    print("\n=== 测试完成 ===") 