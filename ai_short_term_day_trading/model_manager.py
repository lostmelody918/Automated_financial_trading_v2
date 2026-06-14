import os
import glob
import json
import torch
from datetime import datetime

class TradingModelManager:
    """負責 AI 交易模型的版本控制與 Metadata 紀錄"""

    def __init__(self, model_dir="saved_models"):
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def save_model(self, model, optimizer, metrics, hyperparameters):
        version = 1
        while True:
            base_name = f"trading_model_v{version}"
            model_path = os.path.join(self.model_dir, f"{base_name}.pth")
            meta_path = os.path.join(self.model_dir, f"{base_name}_metadata.json")

            if not os.path.exists(model_path) and not os.path.exists(meta_path):
                break
            version += 1

        checkpoint = {
            'version': version,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'timestamp': datetime.now().isoformat()
        }

        metadata = {
            "experiment_info": {
                "version": f"v{version}",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_class": type(model).__name__
            },
            "hyperparameters": hyperparameters,
            "performance": metrics
        }

        try:
            torch.save(checkpoint, model_path)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            print(f"✅ [ModelManager] 成功儲存版本 v{version}")
            print(f"   - 模型檔: {model_path}")
            print(f"   - 紀錄檔: {meta_path}")
        except Exception as e:
            print(f"❌ [ModelManager] 儲存失敗: {str(e)}")

    def load_latest_model(self, model, optimizer=None):
        files = glob.glob(os.path.join(self.model_dir, "*.pth"))
        if not files:
            print("⚠️ [ModelManager] 找不到歷史模型，將使用隨機初始化權重。")
            return model, optimizer, 0

        target_class_name = type(model).__name__
        valid_files = []

        for file in files:
            meta_path = file.replace(".pth", "_metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    if meta.get("experiment_info", {}).get("model_class") == target_class_name:
                        valid_files.append(file)
                except Exception:
                    pass
            else:
                # 為了向下相容，若無 metadata 且原本只有一種模型，可以選擇加入
                # 但為保險起見，如果有不同模型了，不符合的就不載入
                pass

        if not valid_files:
            # 如果沒有找到有對應 metadata 的檔案，試著退回原本找最新檔案的邏輯(向下相容)
            print(f"⚠️ [ModelManager] 找不到明確標記為 {target_class_name} 的模型。")
            # 不過這樣可能會載入錯誤架構導致 crash，所以在此返回預設值
            print("⚠️ [ModelManager] 將使用隨機初始化權重。")
            return model, optimizer, 0

        latest_file = max(valid_files, key=os.path.getctime)
        print(f"📦 [ModelManager] 發現最新模型 ({target_class_name})，準備載入：{latest_file}")

        checkpoint = torch.load(latest_file)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer, checkpoint.get('version', 0)