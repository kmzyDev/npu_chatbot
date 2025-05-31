## 前置き
npuでtext 2 textなモデルを動かすデスクトップソフトです  
https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B を使わせて頂いてます    
Intel Core Ultraシリーズ搭載のPCを想定しています

## このレポジトリをクローンしたらやること
1. **ライブラリのバグ対応（2025年5月現在）**  
uv sync  
.venv/Lib/site-packages/  
intel_npu_acceleration_library/nn/llm.py 245行目付近
```python
# return attn_output, None, past_key_value
return attn_output, past_key_value
```

同268行目付近
```python
# rotary_emb=layer.rotary_emb,
rotary_emb=torch.nn.Module,
```

.venv/Lib/site-packages/intel_npu_acceleration_library/modelling.py 97行目付近  
※2.6以降のtorchを使う場合
```python
# return torch.load(model_path)
return torch.load(model_path, weights_only=False)
```

2. **モデル配置**  
uv run dlmodel.py

3. **起動**  
uv run npu_chatbot  
※モデルロードにつきアプリ起動まで1分程度かかります

**（おまけ1）UI編集のについて**  
.venv/Lib/site-packages/PySide6/designer.exe を使ってfront.uiを編集  
uv run pyside6-uic front.ui -o front.py　でfront.uiからfront.pyが出来上がります 

**（おまけ2）ビルド**  
ビルドもできるようになっています  
uv build  
uv tool install dist\npu_chatbot-0.1.0-py3-none-any.whl  
（.exeファイルから起動する時はassets以下も同一階層に置いてください）
