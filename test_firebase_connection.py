"""
Firebase Firestore连接测试脚本
测试Firestore的增删改查功能
"""

import firebase_admin
from firebase_admin import credentials, firestore

def test_firestore():
    """测试Firestore连接和基本操作"""
    try:
        # 初始化Firebase应用
        cred = credentials.Certificate("firebase-key.json")
        firebase_admin.initialize_app(cred)
        
        print("✅ Firebase连接成功！")
        
        # 获取Firestore客户端
        db = firestore.client()
        
        # 查看所有集合
        print("\n=== 查看所有集合 ===")
        collections = db.collections()
        collection_list = [col.id for col in collections]
        print(f"集合列表: {collection_list}")
        
        if collection_list:
            # 选择第一个集合进行测试
            test_collection = collection_list[0]
            print(f"\n=== 测试集合: {test_collection} ===")
            
            # 读取数据
            print("📖 读取数据...")
            docs = db.collection(test_collection).limit(3).stream()
            for doc in docs:
                print(f"  文档ID: {doc.id}")
                print(f"  数据: {doc.to_dict()}")
            
            # 写入测试数据
            print(f"\n✏️ 写入测试数据到 {test_collection}...")
            test_data = {
                'name': '测试数据',
                'timestamp': firestore.SERVER_TIMESTAMP,
                'value': 123
            }
            doc_ref = db.collection(test_collection).add(test_data)
            print(f"  新文档ID: {doc_ref[1].id}")
            
            # 更新数据
            print(f"\n🔄 更新数据...")
            doc_ref[1].update({'value': 456, 'updated': True})
            print("  数据已更新")
            
            # 删除测试数据
            print(f"\n🗑️ 删除测试数据...")
            doc_ref[1].delete()
            print("  测试数据已删除")
        
        # 清理
        firebase_admin.delete_app(firebase_admin.get_app())
        print("\n✅ 测试完成，连接已关闭")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    print("开始测试Firestore...")
    test_firestore()
