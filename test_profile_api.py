#!/usr/bin/env python3
"""
Profile 与音频生成 API 可配置测试脚本（直接调用 api_server HTTP 接口）

用法示例：

# 1) Profile 管理
python3 test_profile_api.py --url http://localhost:8101 --op list-profiles
python3 test_profile_api.py --url http://localhost:8101 --op get-profile --profile-name male_en_british
python3 test_profile_api.py --url http://localhost:8101 --op create-profile --profile-name my_profile \
  --profile-desc "A friendly male British accent" --characteristics '{"accent":"British","gender":"male"}'
python3 test_profile_api.py --url http://localhost:8101 --op update-profile --profile-name my_profile \
  --profile-desc "Updated description" --characteristics '{"tone":"calm"}'
python3 test_profile_api.py --url http://localhost:8101 --op delete-profile --profile-name my_profile

# 2) 生成音频（直接文本）
python3 test_profile_api.py --url http://localhost:8101 --op generate \
  --text "Hello, this is a test." --save-path ./out.wav --ref-audio profile:male_en_british --scene-prompt quiet_indoor

# 3) 生成音频（文本文件）
python3 test_profile_api.py --url http://localhost:8101 --op generate \
  --text-file ./some_text.txt --save-path ./out.wav --ref-audio en_woman

# 4) 上传音频/场景文件
python3 test_profile_api.py --url http://localhost:8101 --op upload-audio --audio-file ./my_voice.wav
python3 test_profile_api.py --url http://localhost:8101 --op upload-scene --scene-file ./my_scene.txt
"""

import argparse
import base64
import json
import os
from pathlib import Path
import sys
import requests


def save_audio_from_base64(audio_base64: str, output_path: Path) -> bool:
    try:
        audio_bytes = base64.b64decode(audio_base64)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
        print(f"音频已保存到: {output_path}")
        return True
    except Exception as e:
        print(f"保存音频失败: {e}")
        return False


def read_text_file(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding='utf-8').strip()
    except UnicodeDecodeError:
        return file_path.read_text(encoding='gbk').strip()


def op_list_profiles(base_url: str):
    print("=== 获取所有Profile ===")
    resp = requests.get(f"{base_url}/profiles")
    data = resp.json()
    if data.get('success'):
        print(f"Profile数量: {data.get('count', 0)}")
        for p in data.get('profiles', []):
            print(f"  - {p['name']}: {p['description'][:60]}...")
    else:
        print(f"失败: {data.get('message')}")


def op_get_profile(base_url: str, profile_name: str):
    print(f"=== 获取Profile: {profile_name} ===")
    resp = requests.get(f"{base_url}/profiles/{profile_name}")
    data = resp.json()
    if data.get('success'):
        p = data.get('profile')
        print(f"名称: {p['name']}")
        print(f"描述: {p['description']}")
    else:
        print(f"失败: {data.get('message')}")


def op_create_profile(base_url: str, name: str, desc: str, characteristics_str: str | None):
    print(f"=== 创建Profile: {name} ===")
    characteristics = {}
    if characteristics_str:
        try:
            characteristics = json.loads(characteristics_str)
        except json.JSONDecodeError as e:
            print(f"characteristics JSON 解析失败: {e}")
            sys.exit(1)
    payload = {
        'name': name,
        'description': desc,
        'characteristics': characteristics,
    }
    resp = requests.post(f"{base_url}/profiles", json=payload)
    data = resp.json()
    if data.get('success'):
        p = data.get('profile')
        print(f"创建成功: {p['name']} - {p['description']}")
    else:
        print(f"失败: {data.get('message')}")


def op_update_profile(base_url: str, name: str, desc: str | None, characteristics_str: str | None):
    print(f"=== 更新Profile: {name} ===")
    characteristics = None
    if characteristics_str:
        try:
            characteristics = json.loads(characteristics_str)
        except json.JSONDecodeError as e:
            print(f"characteristics JSON 解析失败: {e}")
            sys.exit(1)
    payload = {}
    if desc is not None:
        payload['description'] = desc
    if characteristics is not None:
        payload['characteristics'] = characteristics
    resp = requests.put(f"{base_url}/profiles/{name}", json=payload)
    data = resp.json()
    if data.get('success'):
        p = data.get('profile')
        print(f"更新成功: {p['name']} - {p['description']}")
    else:
        print(f"失败: {data.get('message')}")


def op_delete_profile(base_url: str, name: str):
    print(f"=== 删除Profile: {name} ===")
    resp = requests.delete(f"{base_url}/profiles/{name}")
    data = resp.json()
    if data.get('success'):
        print("删除成功")
    else:
        print(f"失败: {data.get('message')}")


def op_generate(base_url: str, text: str | None, text_file: str | None, ref_audio: str | None,
                scene_prompt: str | None, temperature: float, top_k: int | None, top_p: float,
                max_new_tokens: int, seed: int | None, ras_win_len: int, ras_win_max_num_repeat: int,
                chunk_method: str | None, chunk_max_word_num: int, chunk_max_num_turns: int,
                generation_chunk_buffer_size: int | None, save_path: str):
    print("=== 生成音频 ===")
    if text is None and not text_file:
        print("必须提供 --text 或 --text-file 之一")
        sys.exit(1)
    if text is None and text_file:
        text_path = Path(text_file)
        if not text_path.exists():
            print(f"文本文件不存在: {text_path}")
            sys.exit(1)
        text = read_text_file(text_path)

    payload = {
        'text': text,
        'ref_audio': ref_audio,
        'scene_prompt': scene_prompt,
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
        'max_new_tokens': max_new_tokens,
        'seed': seed,
        'ras_win_len': ras_win_len,
        'ras_win_max_num_repeat': ras_win_max_num_repeat,
        'chunk_method': chunk_method,
        'chunk_max_word_num': chunk_max_word_num,
        'chunk_max_num_turns': chunk_max_num_turns,
        'generation_chunk_buffer_size': generation_chunk_buffer_size,
    }

    # 过滤掉 None
    payload = {k: v for k, v in payload.items() if v is not None}

    resp = requests.post(f"{base_url}/generate", json=payload)
    data = resp.json()
    if data.get('success'):
        print(f"生成成功 | 时长: {data.get('duration', 0):.2f}s | 采样率: {data.get('sampling_rate', 0)}Hz")
        output_path = Path(save_path)
        save_audio_from_base64(data['audio_base64'], output_path)
    else:
        print(f"失败: {data.get('message')}")


def op_upload_audio(base_url: str, audio_file: str):
    print("=== 上传音频文件 ===")
    path = Path(audio_file)
    if not path.exists():
        print(f"文件不存在: {path}")
        sys.exit(1)
    with open(path, 'rb') as f:
        files = {'audio_file': (path.name, f, 'audio/wav')}
        resp = requests.post(f"{base_url}/upload-audio", files=files)
    data = resp.json()
    if data.get('success'):
        print(f"上传成功: {data.get('uploaded_file')}")
        print(f"当前音频总数: {data.get('count', 0)}")
    else:
        print(f"失败: {data.get('message')}")


def op_upload_scene(base_url: str, scene_file: str):
    print("=== 上传场景文本文件 ===")
    path = Path(scene_file)
    if not path.exists():
        print(f"文件不存在: {path}")
        sys.exit(1)
    with open(path, 'rb') as f:
        files = {'scene_file': (path.name, f, 'text/plain')}
        resp = requests.post(f"{base_url}/upload-scene", files=files)
    data = resp.json()
    if data.get('success'):
        print(f"上传成功: {data.get('uploaded_file')}")
        print(f"当前场景总数: {data.get('count', 0)}")
    else:
        print(f"失败: {data.get('message')}")


def main():
    parser = argparse.ArgumentParser(description="Profile与生成API测试（HTTP）")
    parser.add_argument('--url', default='http://localhost:8101', help='API服务器URL')
    parser.add_argument('--op', required=True, choices=[
        'list-profiles', 'get-profile', 'create-profile', 'update-profile', 'delete-profile',
        'generate', 'upload-audio', 'upload-scene',
    ], help='操作类型')

    # Profile 参数
    parser.add_argument('--profile-name', help='Profile名称')
    parser.add_argument('--profile-desc', help='Profile描述')
    parser.add_argument('--characteristics', help='Profile特征(JSON字符串)')

    # 生成音频参数
    parser.add_argument('--text', help='直接提供的文本')
    parser.add_argument('--text-file', help='文本文件路径')
    parser.add_argument('--ref-audio', help='参考音频（如 profile:male_en_british 或 en_woman）')
    parser.add_argument('--scene-prompt', default='quiet_indoor', help='场景描述名（对应服务器配置）')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--ras-win-len', type=int, default=7)
    parser.add_argument('--ras-win-max-num-repeat', type=int, default=2)
    parser.add_argument('--chunk-method', choices=['speaker', 'word'], help='分块方法')
    parser.add_argument('--chunk-max-word-num', type=int, default=200)
    parser.add_argument('--chunk-max-num-turns', type=int, default=1)
    parser.add_argument('--generation-chunk-buffer-size', type=int)
    parser.add_argument('--save-path', default='./generated.wav', help='生成音频保存路径')

    # 上传文件参数
    parser.add_argument('--audio-file', help='要上传的音频文件路径')
    parser.add_argument('--scene-file', help='要上传的场景文本文件路径')

    args = parser.parse_args()

    base_url = args.url

    if args.op == 'list-profiles':
        op_list_profiles(base_url)
    elif args.op == 'get-profile':
        if not args.profile_name:
            print('--profile-name 必填')
            sys.exit(1)
        op_get_profile(base_url, args.profile_name)
    elif args.op == 'create-profile':
        if not args.profile_name or not args.profile_desc:
            print('--profile-name 与 --profile-desc 必填')
            sys.exit(1)
        op_create_profile(base_url, args.profile_name, args.profile_desc, args.characteristics)
    elif args.op == 'update-profile':
        if not args.profile_name:
            print('--profile-name 必填')
            sys.exit(1)
        op_update_profile(base_url, args.profile_name, args.profile_desc, args.characteristics)
    elif args.op == 'delete-profile':
        if not args.profile_name:
            print('--profile-name 必填')
            sys.exit(1)
        op_delete_profile(base_url, args.profile_name)
    elif args.op == 'generate':
        op_generate(
            base_url=base_url,
            text=args.text,
            text_file=args.text_file,
            ref_audio=args.ref_audio,
            scene_prompt=args.scene_prompt,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            ras_win_len=args.ras_win_len,
            ras_win_max_num_repeat=args.ras_win_max_num_repeat,
            chunk_method=args.chunk_method,
            chunk_max_word_num=args.chunk_max_word_num,
            chunk_max_num_turns=args.chunk_max_num_turns,
            generation_chunk_buffer_size=args.generation_chunk_buffer_size,
            save_path=args.save_path,
        )
    elif args.op == 'upload-audio':
        if not args.audio_file:
            print('--audio-file 必填')
            sys.exit(1)
        op_upload_audio(base_url, args.audio_file)
    elif args.op == 'upload-scene':
        if not args.scene_file:
            print('--scene-file 必填')
            sys.exit(1)
        op_upload_scene(base_url, args.scene_file)


if __name__ == '__main__':
    main()
