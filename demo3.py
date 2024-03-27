from openai import OpenAI
import httpx

client = OpenAI(
    base_url="https://oneapi.xty.app/v1",
    api_key="sk....",
    http_client=httpx.Client(
        base_url="https://oneapi.xty.app/v1",
        follow_redirects=True,
    ),
)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
    messages=[
        # 第一条消息，表示系统向用户打招呼。
        {"role": "system", "content": "Hello!"},
        # 第一条消息，表示系统向用户打招呼或提问。
        {"role": "user", "content": "已知：低成本、部署广泛的优点成为了一种热门的定位技术 。根据采集信号的不同 ，基于WiFi的室内定位技术可分为基于接收信号强度 （RSS）和基于信道状态信息 （CSI）[3]两类方法 。RSS作为一种粗粒度信息 ，难以提供准确可靠的信息用于定位 。作为一种细粒度信息 ，CSI可以获取更多的信息来提高定位精度 。通过部署设备获取信道状态信息 ，系"
                                    "根据采集信号的不同，基于WiFi的室内定位技术可分为哪两种？"},
    ]
)

print('123')
print(completion)
