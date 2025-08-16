"""
获取speech_transcripts_offline数据
按group_id和user_id排列并导出为CSV
支持获取全部小组的语音内容
"""
import traceback
import json

import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from datetime import datetime, timedelta, timezone

from google.cloud.firestore_v1 import FieldFilter


def process_single_group(db, group_config):
    """处理单个小组的语音数据"""
    group_key = group_config["group_key"]
    group_id = group_config["group_id"]
    base_time_str = group_config["base_time_str"]
    
    print(f"\n📖 处理小组: {group_key} (ID: {group_id})")
    print(f"   基准时间: {base_time_str}")
    
    # 获取该小组的语音数据
    all_data = [doc.to_dict() for doc in db.collection("speech_transcripts_offline")
                .where(filter=FieldFilter("group_id", "==", group_id))
                .stream()]
    
    print(f"   获取到 {len(all_data)} 条语音记录")
    
    if not all_data:
        print(f"   ⚠️  小组 {group_key} 没有语音数据")
        return None
    
    # 按开始时间排序
    all_data = sorted(all_data, key=lambda x: x.get("start"))
    
    # 解析基准时间
    base_time = datetime.strptime(base_time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    interval = 10  # 秒
    num_intervals = 180  # 30分钟
    
    # 生成区间起止时间
    intervals = [(base_time + timedelta(seconds=i * interval), base_time + timedelta(seconds=(i + 1) * interval))
                 for i in range(num_intervals)]
    
    # 统计结构：{speaker: [每个区间的发言时长]}
    speakers = set(d['speaker'] for d in all_data)
    result = {spk: [0.0] * num_intervals for spk in speakers}
    
    for d in all_data:
        s = datetime.fromisoformat(d['start'])
        e = datetime.fromisoformat(d['end'])
        spk = d['speaker']
        # 跳过不在半小时内的
        if e < base_time or s > base_time + timedelta(seconds=interval * num_intervals):
            continue
        # 计算每个区间的重叠时长
        for idx, (intv_start, intv_end) in enumerate(intervals):
            overlap_start = max(s, intv_start)
            overlap_end = min(e, intv_end)
            overlap = (overlap_end - overlap_start).total_seconds()
            if overlap > 0:
                result[spk][idx] += overlap
    
    # 转为DataFrame并添加group_key列
    df = pd.DataFrame(result).T
    df.columns = [
        f"{(i * interval) // 60:02d}:{(i * interval) % 60:02d}-{((i + 1) * interval) // 60:02d}:{((i + 1) * interval) % 60:02d}"
        for i in range(num_intervals)]
    df.index.name = "speaker"
    df.reset_index(inplace=True)
    df.insert(0, "group_key", group_key)
    
    # 控制数据精度，四舍五入到2位小数
    numeric_columns = df.select_dtypes(include=[float]).columns
    df[numeric_columns] = df[numeric_columns].round(2)
    
    return df


def get_speech_transcripts():
    """获取所有小组的语音转录数据"""
    try:
        # 读取小组配置
        print("📋 读取小组配置文件...")
        with open("all_groups_mapping.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        groups = config["groups_mapping"]
        print(f"✅ 找到 {len(groups)} 个小组")
        
        # 初始化Firebase应用
        cred = credentials.Certificate("firebase-key.json")
        firebase_admin.initialize_app(cred)
        
        print("✅ Firebase连接成功！")
        
        # 获取Firestore客户端
        db = firestore.client()
        
        # 处理所有小组
        all_groups_data = []
        
        for group_config in groups:
            try:
                group_df = process_single_group(db, group_config)
                if group_df is not None:
                    all_groups_data.append(group_df)
            except Exception as e:
                print(f"❌ 处理小组 {group_config['group_key']} 时出错: {str(e)}")
                continue
        
        if not all_groups_data:
            print("❌ 没有获取到任何小组的数据")
            return
        
        # 合并所有小组数据
        print(f"\n🔄 合并 {len(all_groups_data)} 个小组的数据...")
        final_df = pd.concat(all_groups_data, ignore_index=True)
        
        print(f"✅ 最终数据形状: {final_df.shape}")
        print(f"   包含 {final_df['group_key'].nunique()} 个小组")
        print(f"   包含 {final_df['speaker'].nunique()} 个发言者")
        
        # 导出csv
        output_filename = "all_groups_speaker_time_by_10s.csv"
        final_df.to_csv(output_filename, encoding="utf-8-sig", index=False)
        print(f"\n💾 数据已导出到: {output_filename}")
        
        # 清理
        firebase_admin.delete_app(firebase_admin.get_app())
        print("\n✅ 连接已关闭")
        
    except Exception as e:
        print(f"❌ 获取数据失败: {traceback.format_exc()}")


if __name__ == "__main__":
    print("开始获取所有小组的语音转录数据...")
    get_speech_transcripts()
