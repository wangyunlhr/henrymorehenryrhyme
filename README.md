环境, 参考 `docs/INSTALL.md`

---

## 1. Waymo 数据处理代码

脚本：`create_waymo.py`

- `start_id` 是序列编号  
- `clip_len` 表示当前一共处理多少个

示例图：  
<img width="807" height="287" alt="image" src="https://github.com/user-attachments/assets/f49a66ae-788c-4927-8dac-5e347efc22b8" />

---

## 2. KITTI 数据处理代码

脚本：`create_kitti.py`

```bash
python main_kitti.py \
  --seq_list 0000 \
  --start_id 0 \
  --clip_size 20
  
- `seq` 是序列编号  
- 已经设定每 **200 帧** 是一个块  
- `start_id` 表示第一个块编号  
- `clip_size` 表示当前一共处理多少个

示例图： 
<img width="1172" height="375" alt="image" src="https://github.com/user-attachments/assets/10a226d1-d122-48e8-b49a-93f67a4b109a" />
