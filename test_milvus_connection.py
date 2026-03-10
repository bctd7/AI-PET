"""
简单的 Milvus 连接测试脚本。

用法（在虚拟环境中执行）：

    python test_milvus_connection.py

会尝试：
1. 使用当前 Vector_DataBaseConfig 里的 uri / token / db_name 创建 MilvusClient
2. 打印现有数据库列表
3. 如果成功，打印“连接成功”及指定 db_name 是否存在
4. 如果失败，打印异常详情
"""

from pymilvus import MilvusClient

from core.config import Vector_DataBaseConfig


def main() -> None:
    config = Vector_DataBaseConfig()
    print(f"尝试连接 Milvus: uri={config.uri!r}, db_name={config.db_name!r}")

    try:
        client = MilvusClient(uri=config.uri, token=config.token or None)
    except Exception as e:
        print(f"[失败] 创建 MilvusClient 时出错: {e!r}")
        return

    try:
        try:
            dbs = client.list_databases()
            print(f"[成功] 当前数据库列表: {dbs}")
        except Exception as e:
            print(f"[失败] 列出数据库时出错: {e!r}")
            return

        if config.db_name in dbs:
            print(f"[成功] 目标数据库 {config.db_name!r} 已存在。")
        else:
            print(f"[提示] 目标数据库 {config.db_name!r} 不存在，可以由应用在运行时创建。")
    finally:
        client.close()
        print("已关闭 MilvusClient 连接。")


if __name__ == "__main__":
    main()

