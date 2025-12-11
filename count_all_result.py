import os
import json
from collections import defaultdict

def recursive_add(d_sum, d_new):
    """
    递归地将 d_new 中的数值加到 d_sum 上（支持多层嵌套）
    """
    for k, v in d_new.items():
        if isinstance(v, dict):
            if k not in d_sum:
                d_sum[k] = {}
            recursive_add(d_sum[k], v)
        elif isinstance(v, (int, float)):
            d_sum[k] = d_sum.get(k, 0.0) + v

def recursive_divide(d_sum, n):
    """
    递归地除以 n，得到平均值
    """
    d_avg = {}
    for k, v in d_sum.items():
        if isinstance(v, dict):
            d_avg[k] = recursive_divide(v, n)
        else:
            d_avg[k] = v / n
    return d_avg

# ====== 修改这里为你的根目录 ======
root_dir = "/data0/code/LiDAR-RT-main/val/kitti_dropoutmask"   # 里面有多个序列子文件夹，每个子文件夹下有 test/metrics/result_all.json
# ===============================

result_sum = {}
count = 0

for seq_name in os.listdir(root_dir):
    metrics_path = os.path.join(root_dir, seq_name, "test", "metrics", "results_all.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
        recursive_add(result_sum, data)
        count += 1
        print(f"Loaded {seq_name}")

if count == 0:
    print("未找到任何 result_all.json 文件！")
else:
    avg_result = recursive_divide(result_sum, count)
    print(f"\n共统计 {count} 个序列，平均结果：\n")
    print(json.dumps(avg_result, indent=4))

out_path = os.path.join(root_dir, "average_result.json")
with open(out_path, "w") as f:
    json.dump(avg_result, f, indent=4)
print(f"平均结果已保存到: {out_path}")