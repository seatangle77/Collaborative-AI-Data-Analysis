"""
获取note_edit_history数据
按group_key和speaker排列并导出为CSV
支持获取全部小组的编辑历史数据
"""
import traceback
import json
import re

import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from datetime import datetime, timedelta, timezone

from google.cloud.firestore_v1 import FieldFilter


def convert_east8_to_utc(east8_time_str):
    """将东八区时间字符串转换为UTC时间"""
    # 解析东八区时间
    east8_time = datetime.strptime(east8_time_str, "%Y-%m-%d %H:%M:%S")
    # 转换为UTC时间（减去8小时）
    utc_time = east8_time - timedelta(hours=8)
    return utc_time.replace(tzinfo=timezone.utc)


def parse_delta_field(delta_data):
    """解析delta字段，计算字符变化量"""
    if not delta_data:
        return 0
    
    total_chars = 0
    
    # 如果delta是字符串，尝试解析JSON
    if isinstance(delta_data, str):
        try:
            delta_data = json.loads(delta_data)
        except:
            return 0
    
    # 如果delta是字典，遍历操作
    if isinstance(delta_data, dict):
        for key, operation in delta_data.items():
            if isinstance(operation, dict):
                # 处理insert操作
                if 'insert' in operation:
                    insert_text = operation['insert']
                    if isinstance(insert_text, str):
                        total_chars += len(insert_text)
                # 处理delete操作（如果有的话）
                if 'delete' in operation:
                    delete_count = operation['delete']
                    if isinstance(delete_count, int):
                        total_chars += delete_count
    
    return total_chars


def count_edit_records(delta_data):
    """计算编辑记录数量（基于delta字段）"""
    if not delta_data:
        return 0
    
    # 每个delta对象算作一次编辑记录
    return 1


def process_single_group_edits(db, group_config):
    """处理单个小组的编辑历史数据"""
    group_key = group_config["group_key"]
    group_id = group_config["group_id"]
    base_time_str = group_config["base_time_str"]
    group_members = group_config["group_members"]
    
    print(f"\n📝 处理小组编辑历史: {group_key} (ID: {group_id})")
    print(f"   基准时间: {base_time_str}")
    
    # 时区转换：东八区 -> UTC
    base_time_utc = convert_east8_to_utc(base_time_str)
    print(f"   UTC时间: {base_time_utc}")
    
    # 获取该小组所有用户的编辑数据
    all_edit_data = []
    
    # 统计每个userid的记录数
    user_record_counts = {}
    
    for member in group_members:
        user_id = member["user_id"]
        speaker = member["speaker"]
        
        # 查询该用户的编辑记录
        edit_docs = db.collection("note_edit_history").where(
            filter=FieldFilter("userId", "==", user_id)
        ).stream()
        
        user_records = []
        for doc in edit_docs:
            edit_data = doc.to_dict()
            edit_data['speaker'] = speaker
            edit_data['group_key'] = group_key
            all_edit_data.append(edit_data)
            user_records.append(edit_data)
        
        # 记录该用户的记录数
        user_record_counts[speaker] = len(user_records)
        print(f"   👤 {speaker} (user_id: {user_id}): {len(user_records)} 条记录")
    
    print(f"   获取到 {len(all_edit_data)} 条编辑记录")
    
    if not all_edit_data:
        print(f"   ⚠️  小组 {group_key} 没有编辑数据")
        return None
    
    # 按时间排序
    all_edit_data = sorted(all_edit_data, key=lambda x: x.get("updatedAt", ""))
    
    # 设置时间参数
    interval = 10  # 秒
    num_intervals = 180  # 30分钟
    
    # 生成时间区间
    intervals = [(base_time_utc + timedelta(seconds=i * interval), 
                  base_time_utc + timedelta(seconds=(i + 1) * interval))
                 for i in range(num_intervals)]
    
    # 统计结构：{speaker: [每个区间的编辑字符数和编辑次数]}
    result_chars = {member["speaker"]: [None] * num_intervals for member in group_members}
    result_count = {member["speaker"]: [0] * num_intervals for member in group_members}
    
    # 处理每条编辑记录
    for edit_record in all_edit_data:
        updated_at_str = edit_record.get("updatedAt", "")
        if not updated_at_str:
            continue
        
        # 解析UTC时间
        try:
            # 移除Z后缀并解析
            updated_at_str = updated_at_str.replace('Z', '+00:00')
            edit_time = datetime.fromisoformat(updated_at_str)
        except:
            continue
        
        speaker = edit_record.get("speaker", "")
        if not speaker:
            continue
        
        # 跳过不在时间范围内的记录
        if edit_time < base_time_utc or edit_time > base_time_utc + timedelta(seconds=interval * num_intervals):
            continue
        
        # 计算字符数和编辑次数
        char_count = None
        edit_count = 0
        
        # 获取字符数（优先使用charCount字段）
        if "charCount" in edit_record and edit_record["charCount"] is not None:
            try:
                char_count = int(edit_record["charCount"])
            except:
                char_count = None
        
        # 计算编辑次数（基于delta字段）
        if "delta" in edit_record:
            edit_count = count_edit_records(edit_record["delta"])
        
        # 找到对应的时间区间
        for idx, (intv_start, intv_end) in enumerate(intervals):
            if intv_start <= edit_time < intv_end:
                # 填充字符数（有记录时）
                if char_count is not None:
                    result_chars[speaker][idx] = char_count
                # 累加编辑次数
                result_count[speaker][idx] += edit_count
                break
    
    # 创建字符数和编辑次数的DataFrame
    df_chars = pd.DataFrame(result_chars).T
    df_count = pd.DataFrame(result_count).T
    
    # 确保字符数列是整数类型，避免小数点
    for col in df_chars.columns:
        df_chars[col] = pd.to_numeric(df_chars[col], errors='coerce').astype('Int64')
    
    # 生成列名
    time_columns = [
        f"{(i * interval) // 60:02d}:{(i * interval) % 60:02d}-{((i + 1) * interval) // 60:02d}:{((i + 1) * interval) % 60:02d}"
        for i in range(num_intervals)]
    
    # 设置列名
    df_chars.columns = [f"note_edit_chars_{col}" for col in time_columns]
    df_count.columns = [f"note_edit_count_{col}" for col in time_columns]
    
    # 处理索引和speaker列
    df_chars.index.name = "speaker"
    df_count.index.name = "speaker"
    df_chars.reset_index(inplace=True)
    df_count.reset_index(inplace=True)
    
    # 从df_chars中提取speaker列，避免重复
    speaker_col = df_chars["speaker"].copy()
    
    # 删除两个DataFrame中的speaker列，避免合并时重复
    df_chars = df_chars.drop("speaker", axis=1)
    df_count = df_count.drop("speaker", axis=1)
    
    # 合并两个DataFrame
    df = pd.concat([df_chars, df_count], axis=1)
    
    # 重新插入speaker列到最前面
    df.insert(0, "speaker", speaker_col)
    
    # 重新排列列的顺序：group_key, speaker, 然后交替字符数和编辑次数
    cols = ["group_key", "speaker"]
    for i in range(num_intervals):
        cols.append(f"note_edit_chars_{time_columns[i]}")
        cols.append(f"note_edit_count_{time_columns[i]}")
    
    df.insert(0, "group_key", group_key)
    df = df[cols]
    
    return df


def get_note_edit_history():
    """获取所有小组的编辑历史数据"""
    try:
        # 读取小组配置
        print("📋 读取小组配置文件...")
        with open("complete_groups_mapping.json", "r", encoding="utf-8") as f:
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
                group_df = process_single_group_edits(db, group_config)
                if group_df is not None:
                    all_groups_data.append(group_df)
            except Exception as e:
                print(f"❌ 处理小组 {group_config['group_key']} 时出错: {str(e)}")
                traceback.print_exc()
                continue
        
        if not all_groups_data:
            print("❌ 没有获取到任何小组的数据")
            return
        
        # 合并所有小组数据
        print(f"\n🔄 合并 {len(all_groups_data)} 个小组的数据...")
        final_df = pd.concat(all_groups_data, ignore_index=True)
        
        print(f"✅ 最终数据形状: {final_df.shape}")
        print(f"   包含 {final_df['group_key'].nunique()} 个小组")
        print(f"   包含 {final_df['speaker'].nunique()} 个用户")
        
        # 统计每个小组的记录数
        print(f"\n📊 各小组记录数统计:")
        group_counts = final_df['group_key'].value_counts()
        for group_key, count in group_counts.items():
            print(f"   🏷️  {group_key}: {count} 条记录")
        
        # 统计每个用户的记录数
        print(f"\n👥 各用户记录数统计:")
        user_counts = final_df['speaker'].value_counts()
        for speaker, count in user_counts.items():
            print(f"   👤 {speaker}: {count} 条记录")
        
        # 导出csv
        output_filename = "all_groups_note_edit_history_by_10s.csv"
        final_df.to_csv(output_filename, encoding="utf-8-sig", index=False)
        print(f"\n💾 数据已导出到: {output_filename}")
        
        # 清理
        firebase_admin.delete_app(firebase_admin.get_app())
        print("\n✅ 连接已关闭")
        
    except Exception as e:
        print(f"❌ 获取数据失败: {traceback.format_exc()}")


if __name__ == "__main__":
    print("开始获取所有小组的编辑历史数据...")
    get_note_edit_history()
