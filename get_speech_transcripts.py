"""
获取speech_transcripts_offline数据
按group_id和user_id排列并导出为CSV
"""

import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from datetime import datetime

def get_speech_transcripts():
    """获取语音转录数据"""
    try:
        # 初始化Firebase应用
        cred = credentials.Certificate("firebase-key.json")
        firebase_admin.initialize_app(cred)
        
        print("✅ Firebase连接成功！")
        
        # 获取Firestore客户端
        db = firestore.client()
        
        print("\n📖 读取 speech_transcripts_offline 集合...")
        
        # 读取speech_transcripts_offline集合中的所有文档
        docs = db.collection("speech_transcripts_offline").stream()
        
        all_data = []
        for doc in docs:
            data = doc.to_dict()
            data['_document_id'] = doc.id
            all_data.append(data)
        
        if all_data:
            # 转换为DataFrame
            df = pd.DataFrame(all_data)
            
            print(f"📊 获取到 {len(df)} 条语音转录数据")
            print(f"📋 数据列: {list(df.columns)}")
            
            # 按group_id和user_id排序
            if 'group_id' in df.columns and 'user_id' in df.columns:
                df_sorted = df.sort_values(['group_id', 'user_id', 'start'])
                print("\n🔄 数据已按 group_id, user_id, start 排序")
            else:
                df_sorted = df
                print("\n⚠️ 未找到 group_id 或 user_id 列，使用原始顺序")
            
            # 显示排序后的前几行
            print(f"\n📖 排序后数据预览:")
            print(df_sorted.head(10))
            
            # 导出为CSV
            csv_filename = 'speech_transcripts_sorted.csv'
            df_sorted.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            print(f"\n✅ 数据已导出到: {csv_filename}")
            
            # 显示一些统计信息
            if 'group_id' in df.columns:
                print(f"\n📊 统计信息:")
                print(f"  总组数: {df['group_id'].nunique()}")
                print(f"  总用户数: {df['user_id'].nunique()}")
                print(f"  总转录条数: {len(df)}")
                
                # 按组统计
                group_stats = df.groupby('group_id').agg({
                    'user_id': 'nunique',
                    '_document_id': 'count'
                }).rename(columns={'user_id': '用户数', '_document_id': '转录条数'})
                
                print(f"\n📈 各组统计:")
                print(group_stats.head(10))
            
        else:
            print("ℹ️ 没有找到语音转录数据")
        
        # 清理
        firebase_admin.delete_app(firebase_admin.get_app())
        print("\n✅ 连接已关闭")
        
    except Exception as e:
        print(f"❌ 获取数据失败: {e}")

if __name__ == "__main__":
    print("开始获取语音转录数据...")
    get_speech_transcripts()
