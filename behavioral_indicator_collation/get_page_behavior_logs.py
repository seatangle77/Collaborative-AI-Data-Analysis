#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get Page Behavior Logs
获取并分析pageBehaviorLogs数据，生成类似语音数据的CSV文件
"""

import json
import csv
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any
import os
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import FieldFilter

def convert_east8_to_utc(east8_time_str):
    """将东八区时间字符串转换为UTC时间"""
    # 解析东八区时间
    east8_time = datetime.strptime(east8_time_str, "%Y-%m-%d %H:%M:%S")
    # 转换为UTC时间（减去8小时）
    utc_time = east8_time - timedelta(hours=8)
    return utc_time.replace(tzinfo=timezone.utc)

class PageBehaviorAnalyzer:
    def __init__(self, mapping_file: str, output_file: str):
        """
        初始化分析器
        
        Args:
            mapping_file: complete_groups_mapping.json文件路径
            output_file: 输出CSV文件路径
        """
        self.mapping_file = mapping_file
        self.output_file = output_file
        self.groups_mapping = {}
        self.user_to_group = {}
        self.time_columns = []
        
    def load_groups_mapping(self):
        """加载组映射信息"""
        try:
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.groups_mapping = data.get('groups_mapping', [])
                
            print(f"\n📋 开始加载组映射信息...")
            print(f"   配置文件: {self.mapping_file}")
            print(f"   总组数: {len(self.groups_mapping)}")
            
            # 建立用户ID到组信息的映射
            for i, group in enumerate(self.groups_mapping):
                group_key = group['group_key']
                base_time_str = group['base_time_str']
                
                print(f"\n🔍 正在处理第 {i+1}/{len(self.groups_mapping)} 个组: {group_key}")
                print(f"   原始时间: {base_time_str}")
                
                # 解析组开始时间（东八区）并转换为UTC
                try:
                    base_time_utc = convert_east8_to_utc(base_time_str)
                    group['base_time'] = base_time_utc
                    print(f"   UTC时间: {base_time_utc}")
                except ValueError as e:
                    print(f"   ❌ 解析组 {group_key} 的开始时间失败: {e}")
                    continue
                
                # 显示组成员信息
                group_members = group.get('group_members', [])
                print(f"   成员数量: {len(group_members)}")
                
                # 建立用户映射
                for j, member in enumerate(group_members):
                    user_id = member['user_id']
                    speaker = member['speaker']
                    self.user_to_group[user_id] = {
                        'group_key': group_key,
                        'speaker': speaker,
                        'base_time': base_time_utc
                    }
                    print(f"     - 成员 {j+1}/{len(group_members)}: 用户ID {user_id[:8]}... (发言者: {speaker})")
                    
            print(f"\n✅ 成功加载 {len(self.groups_mapping)} 个组，{len(self.user_to_group)} 个用户")
            print(f"   用户映射表大小: {len(self.user_to_group)}")
            
        except Exception as e:
            print(f"❌ 加载组映射文件失败: {e}")
            raise
    
    def generate_time_columns(self):
        """生成30分钟的时间列（每10秒一个间隔）"""
        self.time_columns = []
        for i in range(180):  # 30分钟 = 180个10秒间隔
            start_min = i // 6
            start_sec = (i % 6) * 10
            end_min = (i + 1) // 6
            end_sec = ((i + 1) % 6) * 10
            
            if end_sec == 0:
                end_sec = 60
                end_min -= 1
                
            time_col = f"{start_min:02d}:{start_sec:02d}-{end_min:02d}:{end_sec:02d}"
            self.time_columns.append(time_col)
        
        print(f"生成 {len(self.time_columns)} 个时间列")
    
    def parse_utc_time(self, utc_str: str) -> datetime:
        """解析UTC时间字符串"""
        try:
            # 移除Z后缀并添加+00:00时区标识
            utc_str_clean = utc_str.replace('Z', '+00:00')
            utc_time = datetime.fromisoformat(utc_str_clean)
            return utc_time
        except Exception as e:
            print(f"解析时间失败 {utc_str}: {e}")
            return None
    
    def calculate_time_interval(self, behavior_time: datetime, group_base_time: datetime) -> int:
        """
        计算行为时间相对于组开始时间的10秒间隔索引
        
        Args:
            behavior_time: 行为时间（UTC）
            group_base_time: 组开始时间（UTC）
            
        Returns:
            10秒间隔的索引（0-179），如果超出30分钟范围则返回-1
        """
        time_diff = behavior_time - group_base_time
        seconds_diff = time_diff.total_seconds()
        
        # 检查是否在30分钟范围内
        if 0 <= seconds_diff <= 1800:  # 30分钟 = 1800秒
            interval_index = int(seconds_diff / 10)
            return min(interval_index, 179)  # 确保不超过179
        else:
            return -1  # 超出范围
    
    def process_time_alignment(self, behavior_data: List[Dict]) -> List[Dict]:
        """
        处理时间对齐逻辑
        
        Args:
            behavior_data: 原始pageBehaviorLogs数据
            
        Returns:
            添加了时间对齐信息的数据
        """
        print(f"\n🔄 开始时间对齐处理...")
        print(f"   原始数据条数: {len(behavior_data)}")
        
        aligned_data = []
        skipped_user_not_found = 0
        skipped_no_window_start = 0
        skipped_time_parse_failed = 0
        skipped_out_of_range = 0
        
        # 统计每个用户的记录数
        user_record_counts = {}
        group_record_counts = {}
        
        for i, record in enumerate(behavior_data):
            if i < 5:  # 只打印前5条记录的详细信息
                print(f"\n   记录 {i+1}:")
                print(f"     userId: {record.get('userId', 'N/A')}")
                print(f"     windowStart: {record.get('windowStart', 'N/A')}")
                print(f"     behaviorData keys: {list(record.get('behaviorData', {}).keys())}")
            
            user_id = record.get('userId')
            
            # 统计每个用户的记录数
            if user_id:
                user_record_counts[user_id] = user_record_counts.get(user_id, 0) + 1
                
                # 如果用户不在映射表中，统计到组"未映射"
                if user_id not in self.user_to_group:
                    group_key = "未映射"
                else:
                    group_key = self.user_to_group[user_id]['group_key']
                    # 统计每个组的记录数
                    group_record_counts[group_key] = group_record_counts.get(group_key, 0) + 1
            
            if user_id not in self.user_to_group:
                skipped_user_not_found += 1
                if i < 5:
                    print(f"     ❌ 用户ID不在映射表中")
                continue
                
            window_start = record.get('windowStart')
            if not window_start:
                skipped_no_window_start += 1
                if i < 5:
                    print(f"     ❌ 缺少windowStart字段")
                continue
                
            # 解析时间
            behavior_time = self.parse_utc_time(window_start)
            if not behavior_time:
                skipped_time_parse_failed += 1
                if i < 5:
                    print(f"     ❌ 时间解析失败")
                continue
                
            # 获取组信息
            group_info = self.user_to_group[user_id]
            group_base_time = group_info['base_time']
            
            if i < 5:
                print(f"     ✅ 用户匹配成功: {group_info['group_key']} - {group_info['speaker']}")
                print(f"     行为时间: {behavior_time}")
                print(f"     组基准时间: {group_base_time}")
            
            # 计算时间间隔索引
            interval_index = self.calculate_time_interval(behavior_time, group_base_time)
            if interval_index == -1:
                skipped_out_of_range += 1
                if i < 5:
                    print(f"     ❌ 时间超出30分钟范围")
                continue
                
            if i < 5:
                print(f"     ✅ 时间对齐成功: 间隔索引 {interval_index}")
            
            # 添加时间对齐信息
            aligned_record = record.copy()
            aligned_record['interval_index'] = interval_index
            aligned_record['group_info'] = group_info
            aligned_data.append(aligned_record)
        
        # 打印用户记录数统计
        print(f"\n📊 用户记录数统计:")
        print(f"   总用户数: {len(user_record_counts)}")
        print(f"   前10个用户的记录数:")
        sorted_users = sorted(user_record_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (user_id, count) in enumerate(sorted_users[:10]):
            if user_id in self.user_to_group:
                group_info = self.user_to_group[user_id]
                print(f"     {i+1}. 用户 {user_id[:8]}... ({group_info['group_key']}-{group_info['speaker']}): {count} 条记录")
            else:
                print(f"     {i+1}. 用户 {user_id[:8]}... (未映射): {count} 条记录")
        
        # 打印组记录数统计
        print(f"\n📊 组记录数统计:")
        print(f"   总组数: {len(group_record_counts)}")
        print(f"   各组记录数:")
        sorted_groups = sorted(group_record_counts.items(), key=lambda x: x[1], reverse=True)
        for group_key, count in sorted_groups:
            print(f"     {group_key}: {count} 条记录")
        
        print(f"\n📊 时间对齐统计:")
        print(f"   原始数据: {len(behavior_data)} 条")
        print(f"   用户ID不匹配: {skipped_user_not_found} 条")
        print(f"   缺少windowStart: {skipped_no_window_start} 条")
        print(f"   时间解析失败: {skipped_time_parse_failed} 条")
        print(f"   时间超出范围: {skipped_out_of_range} 条")
        print(f"   成功对齐: {len(aligned_data)} 条")
        
        return aligned_data
    
    def aggregate_behavior_data(self, aligned_data: List[Dict]) -> Dict[str, List[int]]:
        """
        聚合行为数据到10秒间隔
        
        Args:
            aligned_data: 时间对齐后的数据
            
        Returns:
            用户ID到时间间隔数据的映射
        """
        print(f"\n📈 开始数据聚合...")
        print(f"   对齐后数据条数: {len(aligned_data)}")
        
        user_data = {}
        
        # 初始化所有用户的数据结构
        for user_id in self.user_to_group:
            user_data[user_id] = [0] * 180  # 180个10秒间隔
        
        print(f"   初始化了 {len(user_data)} 个用户的数据结构")
        
        # 统计信息
        total_actions = 0
        users_with_actions = set()
        
        # 填充数据
        for i, record in enumerate(aligned_data):
            user_id = record.get('userId')
            interval_index = record.get('interval_index')
            
            # 计算mouse_action_count
            behavior_logs = record.get('behaviorData', {}).get('tabBehaviorLogs', [])
            action_count = len(behavior_logs)
            
            if i < 5:  # 打印前5条记录的详细信息
                print(f"\n   聚合记录 {i+1}:")
                print(f"     userId: {user_id}")
                print(f"     间隔索引: {interval_index}")
                print(f"     tabBehaviorLogs长度: {action_count}")
                print(f"     behaviorData keys: {list(record.get('behaviorData', {}).keys())}")
            
            # 填充到对应的时间间隔
            user_data[user_id][interval_index] = action_count
            
            total_actions += action_count
            if action_count > 0:
                users_with_actions.add(user_id)
        
        # 统计每个用户的非零间隔数量
        non_zero_intervals = {}
        for user_id, time_data in user_data.items():
            non_zero_count = sum(1 for count in time_data if count > 0)
            non_zero_intervals[user_id] = non_zero_count
        
        print(f"\n📊 数据聚合统计:")
        print(f"   总行为次数: {total_actions}")
        print(f"   有行为的用户数: {len(users_with_actions)}")
        print(f"   用户行为分布:")
        for user_id, non_zero_count in list(non_zero_intervals.items())[:5]:  # 只显示前5个用户
            group_info = self.user_to_group.get(user_id, {})
            group_key = group_info.get('group_key', 'N/A')
            speaker = group_info.get('speaker', 'N/A')
            print(f"     {user_id[:8]}... ({group_key}-{speaker}): {non_zero_count} 个非零间隔")
        
        return user_data
    
    def generate_csv(self, user_data: Dict[str, List[int]]):
        """生成CSV文件"""
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
                # 写入表头
                fieldnames = ['group_key', 'speaker'] + self.time_columns
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # 写入数据
                for user_id, time_data in user_data.items():
                    if user_id not in self.user_to_group:
                        continue
                        
                    group_info = self.user_to_group[user_id]
                    row = {
                        'group_key': group_info['group_key'],
                        'speaker': group_info['speaker']
                    }
                    
                    # 添加时间列数据
                    for i, count in enumerate(time_data):
                        row[self.time_columns[i]] = count
                    
                    writer.writerow(row)
                    
            print(f"成功生成CSV文件: {self.output_file}")
            
        except Exception as e:
            print(f"生成CSV文件失败: {e}")
            raise
    
    def analyze_and_export(self, behavior_data: List[Dict]):
        """主分析流程"""
        print("开始分析pageBehaviorLogs数据...")
        
        # 加载组映射
        self.load_groups_mapping()
        
        # 生成时间列
        self.generate_time_columns()
        
        # 时间对齐处理
        aligned_data = self.process_time_alignment(behavior_data)
        
        # 数据聚合
        user_data = self.aggregate_behavior_data(aligned_data)
        
        # 生成CSV文件
        self.generate_csv(user_data)
        
        print("分析完成！")
        
        # 返回统计信息
        return {
            'total_groups': len(self.groups_mapping),
            'total_users': len(user_data),
            'time_intervals': len(self.time_columns),
            'output_file': self.output_file
        }

def get_page_behavior_logs():
    """从Firebase获取pageBehaviorLogs数据，按组逐个查询"""
    try:
        # 初始化Firebase应用
        cred = credentials.Certificate("firebase-key.json")
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        
        print("✅ Firebase连接成功！")
        
        # 先加载组映射信息
        mapping_file = "complete_groups_mapping.json"
        with open(mapping_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            groups_mapping = data.get('groups_mapping', [])
        
        print(f"\n📋 加载了 {len(groups_mapping)} 个组的信息")
        
        all_behavior_data = []
        group_records = {}  # 每个组的记录数
        user_records = {}   # 每个用户的记录数
        
        # 按组逐个查询
        for i, group in enumerate(groups_mapping):
            group_key = group['group_key']
            group_members = group.get('group_members', [])
            base_time_str = group.get('base_time_str', 'N/A')
            
            # 解析组开始时间（东八区）并转换为UTC
            try:
                base_time_utc = convert_east8_to_utc(base_time_str)
                group['base_time'] = base_time_utc
                utc_time_str = base_time_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
            except ValueError as e:
                print(f"   ❌ 解析组 {group_key} 的开始时间失败: {e}")
                utc_time_str = "时间解析失败"
                continue
            
            print(f"\n🔍 正在查询第 {i+1}/{len(groups_mapping)} 个组: {group_key}")
            print(f"   组开始时间: {base_time_str} (东八区) → {utc_time_str}")
            print(f"   组成员数: {len(group_members)}")
            
            group_total_records = 0
            group_earliest_window_start = None  # 记录该组最早的windowStart
            
            # 查询这个组每个成员的pageBehaviorLogs
            for j, member in enumerate(group_members):
                user_id = member['user_id']
                speaker = member['speaker']
                
                print(f"     - 查询成员 {j+1}/{len(group_members)}: {user_id[:8]}... ({speaker})")
                
                # 查询这个用户的pageBehaviorLogs
                try:
                    docs = db.collection("pageBehaviorLogs").where("userId", "==", user_id).stream()
                    user_data = []
                    for doc in docs:
                        behavior_data = doc.to_dict()
                        user_data.append(behavior_data)
                        
                        # 检查windowStart时间，找到最早的
                        window_start = behavior_data.get('windowStart')
                        if window_start:
                            if group_earliest_window_start is None or window_start < group_earliest_window_start:
                                group_earliest_window_start = window_start
                    
                    user_records[user_id] = len(user_data)
                    group_total_records += len(user_data)
                    
                    print(f"       找到 {len(user_data)} 条记录")
                    
                    # 添加到总数据中
                    all_behavior_data.extend(user_data)
                    
                except Exception as e:
                    print(f"       查询用户 {user_id[:8]}... 失败: {e}")
                    user_records[user_id] = 0
            
            # 记录这个组的总记录数
            group_records[group_key] = group_total_records
            
            # 打印该组最早的windowStart时间
            if group_earliest_window_start:
                print(f"   📅 该组最早的windowStart: {group_earliest_window_start}")
            else:
                print(f"   📅 该组没有找到windowStart数据")
                
            print(f"   ✅ 组 {group_key} 查询完成，总计 {group_total_records} 条记录")
        
        print(f"\n✅ 所有组查询完成！")
        print(f"   总记录数: {len(all_behavior_data)}")
        
        # 打印组记录数统计
        print(f"\n📊 各组记录数统计:")
        for group_key, count in group_records.items():
            print(f"   {group_key}: {count} 条记录")
        
        # 打印用户记录数统计
        print(f"\n📊 各用户记录数统计:")
        sorted_users = sorted(user_records.items(), key=lambda x: x[1], reverse=True)
        for i, (user_id, count) in enumerate(sorted_users):
            # 找到用户所属的组
            user_group = "未知"
            user_speaker = "未知"
            for group in groups_mapping:
                for member in group.get('group_members', []):
                    if member['user_id'] == user_id:
                        user_group = group['group_key']
                        user_speaker = member['speaker']
                        break
                if user_group != "未知":
                    break
            
            print(f"   {i+1:2d}. 用户 {user_id[:8]}... ({user_group}-{user_speaker}): {count:3d} 条记录")
        
        # 清理连接
        firebase_admin.delete_app(firebase_admin.get_app())
        print("✅ Firebase连接已关闭")
        
        return all_behavior_data
        
    except Exception as e:
        print(f"❌ 获取pageBehaviorLogs数据失败: {e}")
        return []

def main():
    """主函数"""
    # 文件路径
    mapping_file = "complete_groups_mapping.json"
    output_file = "all_groups_page_behavior_by_10s.csv"
    
    # 检查文件是否存在
    if not os.path.exists(mapping_file):
        print(f"错误：找不到文件 {mapping_file}")
        return
    
    # 创建分析器
    analyzer = PageBehaviorAnalyzer(mapping_file, output_file)
    
    # 获取pageBehaviorLogs数据
    behavior_data = get_page_behavior_logs()
    
    if not behavior_data:
        print("错误：没有获取到pageBehaviorLogs数据")
        return
    
    try:
        # 执行分析
        result = analyzer.analyze_and_export(behavior_data)
        print(f"分析结果: {result}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")

if __name__ == "__main__":
    main()
