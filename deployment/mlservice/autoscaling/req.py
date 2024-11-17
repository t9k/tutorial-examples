import requests
import time
from concurrent.futures import ThreadPoolExecutor
import subprocess

# 使用 kubectl 命令获取 ENDPOINT
def get_endpoint():
    cmd = "kubectl get mls qwen-2-5-7b -ojsonpath='{.status.address.url}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Failed to get ENDPOINT: {result.stderr}")
    return result.stdout.strip()

# 定义请求函数
def send_request(process_id, endpoint):
    while True:
        try:
            response = requests.post(
                f"{endpoint}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "qwen",
                    "messages": [{"role": "user", "content": "I cannot sleep well, can you give me some advice?"}],
                    "temperature": 0.5
                }
            )
            print(f"Process {process_id} sent a request. Status code: {response.status_code}")
        except Exception as e:
            print(f"Process {process_id} encountered an error: {str(e)}")
        
        time.sleep(1)  # 每次请求之间暂停1秒,避免过于频繁的请求

# 主函数
def main():
    try:
        endpoint = get_endpoint()
        print(f"Using ENDPOINT: {endpoint}")

        # 创建一个包含20个线程的线程池
        with ThreadPoolExecutor(max_workers=20) as executor:
            # 提交20个任务到线程池
            for i in range(1, 21):
                executor.submit(send_request, i, endpoint)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
