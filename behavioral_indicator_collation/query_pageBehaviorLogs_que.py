    
import traceback
import json

import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from datetime import datetime, timedelta, timezone

from google.cloud.firestore_v1 import FieldFilter

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
        

        # 查询3.1: group_id匹配且windowStart在时间范围内的记录
        logs_start = []
        updated_count = 0  # 记录更新数量
        batch = db.batch()  # 创建批量写入操作
        docs_to_update = []  # 存储需要更新的文档引用
        
        query_start = db.collection("pageBehaviorLogs")\
            .where(filter=FieldFilter("windowStart", ">=", "2025-08-03T03:14:55.000Z"))\
            .where(filter=FieldFilter("windowStart", "<=", "2025-08-03T03:44:55.000Z"))\
            .stream()
        l = 0
        for doc in query_start:
            doc_data = doc.to_dict()
            logs_start.append(doc_data)
            print(doc_data['windowStart'])
            print(doc_data['userId'])
            l += 1
            print(l)
            
            # 检查并准备更新unknown的userId
            if doc_data['userId'] == 'lcTy9qaGR6gpZ6QIFxUKkjhE4Jt1':
                batch.update(doc.reference, {'userId': 'pId4CeRGylfZYPwU6rJa87maJEh1'})
                docs_to_update.append(doc.reference)  # 保存文档引用
                updated_count += 1
                print(f"✅ 准备更新记录 {doc.id} 的userId从 'lcTy9qaGR6gpZ6QIFxUKkjhE4Jt1' 到 'pId4CeRGylfZYPwU6rJa87maJEh1'")
        
        # 执行批量更新
        if updated_count > 0:
            try:
                batch.commit()
                print(f"✅ 批量更新成功！共更新了 {updated_count} 条记录")
            except Exception as e:
                print(f"❌ 批量更新失败: {e}")
                print("尝试使用单个更新方式...")
                # 如果批量更新失败，尝试单个更新
                for doc_ref in docs_to_update:
                    try:
                        doc_ref.update({'userId': 'pId4CeRGylfZYPwU6rJa87maJEh1'})
                        print(f"✅ 单个更新成功: {doc_ref.id}")
                    except Exception as single_error:
                        print(f"❌ 单个更新失败 {doc_ref.id}: {single_error}")
        else:
            print("📊 没有找到需要更新的记录")
        
        print(f"\n📊 更新统计: 共更新了 {updated_count} 条记录的userId")
        
        # 清理
        firebase_admin.delete_app(firebase_admin.get_app())
        print("\n✅ 连接已关闭")
        
    except Exception as e:
        print(f"❌ 获取数据失败: {traceback.format_exc()}")


if __name__ == "__main__":
    print("..")
    get_speech_transcripts()