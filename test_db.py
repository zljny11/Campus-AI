import pymysql
try:
    conn = pymysql.connect(
        host='127.0.0.1', # 强制使用 IP 避开 localhost 解析问题
        user='root',
        password='MyPassword!',
        database='uniticketAI'
    )
    print("连接成功！")
    conn.close()
except Exception as e:
    print(f"连接失败: {e}")