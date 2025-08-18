#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询数据库中userId为"unknown"的pageBehaviorLogs数据
按windowStart降序排列并输出到CSV文件
"""

import csv
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import FieldFilter
from datetime import datetime
import os

def query_unknown_user_page_behavior():
    """查询userId为'unknown'且windowStart大于指定时间的pageBehaviorLogs数据"""
    try:
        # 初始化Firebase应用
        cred = credentials.Certificate("firebase-key.json")
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        
        print("✅ Firebase连接成功！")
        
        # 设置查询条件
        target_time = "2025-07-30T06:23:50.000Z"
        print(f"🔍 查询条件:")
        print(f"   userId: 'unknown'")
        print(f"   windowStart > {target_time}")
        
        # 查询数据
        print("\n📖 正在查询数据...")
        
        # 使用复合查询：userId == "unknown" AND windowStart > target_time
        query = db.collection("pageBehaviorLogs").where(
            "userId", "==", "unknown"
        ).where(
            "windowStart", ">", target_time
        ).order_by("windowStart", direction=firestore.Query.DESCENDING)
        
        # 执行查询
        docs = query.stream()
        
        # 收集数据
        behavior_data = []
        for doc in docs:
            data = doc.to_dict()
            data['document_id'] = doc.id  # 添加文档ID
            behavior_data.append(data)
        
        print(f"✅ 查询完成！找到 {len(behavior_data)} 条记录")
        
        # 按windowStart降序排列（虽然查询已经排序，但为了确保，再次排序）
        behavior_data.sort(key=lambda x: x.get('windowStart', ''), reverse=True)
        
        # 输出到CSV文件
        output_file = "unknown_user_page_behavior.csv"
        export_to_csv(behavior_data, output_file)
        
        # 清理连接
        firebase_admin.delete_app(firebase_admin.get_app())
        print("✅ Firebase连接已关闭")
        
        return behavior_data
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        return []

def export_to_csv(data, output_file):
    """将数据导出到CSV文件"""
    try:
        if not data:
            print("⚠️ 没有数据需要导出")
            return
        
        # 获取所有可能的字段名
        all_fields = set()
        for record in data:
            all_fields.update(record.keys())
        
        # 确保重要字段在前面
        field_order = ['document_id', 'userId', 'windowStart', 'windowEnd']
        remaining_fields = [field for field in sorted(all_fields) if field not in field_order]
        fieldnames = field_order + remaining_fields
        
        print(f"\n📝 正在导出数据到 {output_file}...")
        print(f"   总记录数: {len(data)}")
        print(f"   字段数: {len(fieldnames)}")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # 写入数据
            for i, record in enumerate(data):
                # 确保所有字段都有值，缺失的设为空字符串
                row = {field: record.get(field, '') for field in fieldnames}
                writer.writerow(row)
                
                # 打印前几条记录的详细信息
                if i < 3:
                    print(f"\n   记录 {i+1}:")
                    print(f"     document_id: {row.get('document_id', 'N/A')}")
                    print(f"     userId: {row.get('userId', 'N/A')}")
                    print(f"     windowStart: {row.get('windowStart', 'N/A')}")
                    print(f"     windowEnd: {row.get('windowEnd', 'N/A')}")
                    
                    # 显示behaviorData的键（如果存在）
                    behavior_data = record.get('behaviorData', {})
                    if behavior_data:
                        print(f"     behaviorData keys: {list(behavior_data.keys())}")
        
        print(f"\n✅ 成功导出 {len(data)} 条记录到 {output_file}")
        
        # 显示文件大小
        file_size = os.path.getsize(output_file)
        print(f"   文件大小: {file_size / 1024:.2f} KB")
        
    except Exception as e:
        print(f"❌ 导出CSV文件失败: {e}")
        raise

def main():
    """主函数"""
    print("🚀 开始查询userId为'unknown'的pageBehaviorLogs数据...")
    
    # 查询数据
    behavior_data = query_unknown_user_page_behavior()
    
    if behavior_data:
        print(f"\n✅ 查询和导出完成！")
        print(f"   输出文件: unknown_user_page_behavior.csv")
    else:
        print("❌ 没有找到符合条件的数据")

if __name__ == "__main__":
    main()
