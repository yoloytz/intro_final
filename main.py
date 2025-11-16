import torch
print(torch.__version__)  # 确认 >= 2.6

from cnsenti import Emotion
from cnsenti import Sentiment
emotion = Emotion()
test_text = '我好开心啊，非常非常非常高兴！今天我得了一百分，我很兴奋开心，愉快，开心'
result = emotion.emotion_count(test_text)
print(result)

senti = Sentiment()
test_text= '我好开心啊，非常非常非常高兴！今天我得了一百分，我很兴奋开心，愉快，开心'
result = senti.sentiment_count(test_text)
print(result)

from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-binary-chinese")
result = classifier("我好开心啊，非常非常非常高兴！今天我得了一百分，我很兴奋开心，愉快，开心")
print(result)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,        # 16bit 加速推理
    device_map="auto",                # 自动把模型放到 GPU
    trust_remote_code=True
)

def llm_sentiment(text):
    prompt = f"""
请判断下面文本的情感倾向（正向 / 负向 / 中性），并给出一句理由：

文本：{text}
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False  # 不随机，保证可复现性
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

text = "我今天和对象吵了一天架，我很难过"
result = llm_sentiment(text)
print(result)



test_texts = [
    # === 1. 明显正向（8 条）===
    "今天得了满分，我开心到飞起来！",
    "和朋友聚会太愉快了，整个人都被治愈了。",
    "终于完成论文，心情特别轻松。",
    "老板夸我做得好，我真的很激动！",
    "天气真好，出去散步心情特别棒。",
    "收到了面试通过的消息，太开心了！",
    "家里做了我最爱吃的菜，幸福感爆棚。",
    "跑步状态超好，感觉整个人都充满力量。",

    # === 2. 明显负向（8 条）===
    "我真的累坏了，整个人一点力气都没有。",
    "今天被骂得很惨，情绪直接跌到谷底。",
    "失眠了一整夜，难受得想哭。",
    "事情完全不顺利，我心态炸裂。",
    "又失败了，我真的很沮丧。",
    "想到还有一堆任务就头皮发麻。",
    "感觉自己被忽视了，心里很难受。",
    "今天特别不开心，谁都不想说话。",

    # === 3. 反讽文本（8 条）===
    "哇真棒，日程又被临时加满了。",
    "太好了，又要加班到半夜，开心得不得了。",
    "真是优秀啊，项目上线当天服务器就崩。",
    "完美，我的电脑在关键时候自动重启了。",
    "厉害了，又遇见堵车大魔王，真是幸福。",
    "太棒了，快递又双叒叕延误了。",
    "呵呵，考试题目全都不会，真是惊喜。",
    "好家伙，手机刚买一周就摔坏了，真幸运。",

    # === 4. 混合情绪（8 条）===
    "虽然今天很累，但看到结果还是挺欣慰的。",
    "做得不完美，可至少迈出第一步了。",
    "心里有点烦，但收到你的消息还是挺暖心的。",
    "遇到不少困难，不过最终还是解决了。",
    "过程很煎熬，不过学到了很多东西。",
    "有些遗憾，但总算结束了。",
    "还是有点担心，但感觉事情在变好。",
    "虽然被拒绝了，但我觉得自己成长了。",

    # === 5. 多句矛盾/情绪转折（8 条）===
    "一开始特别期待，但后来剧情让我很失望。",
    "刚开始很难过，后面慢慢放下了。",
    "原本挺生气的，结果被你一句话逗笑了。",
    "起初很顺利，可后面的问题越来越多。",
    "前半天状态很好，下午整个人突然情绪低落。",
    "刚收到坏消息，但同时又来了个好消息。",
    "当时特别着急，现在想想其实也没什么。",
    "本来挺害怕，后来发现也还好。",

    # === 6. 真实领域文本（10 条：JD 评论 / 微博 / 聊天）===
    "包装还不错，但物流太慢了，有点不爽。",
    "客服态度很好，可惜产品一般般。",
    "味道不错，就是份量少了点。",
    "手机性能挺强，但发热有点严重。",
    "微博被刷屏了，信息太杂有点烦。",
    "今天地铁挤到不行，但看到小猫视频治愈了一下。",
    "这课讲得挺清楚的，就是作业太多了。",
    "聊天聊着聊着突然被冷场，心情有点怪。",
    "最近天气忽冷忽热，状态不是很好。",
    "本想休息，结果被电话叫去开会。"
]
