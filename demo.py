import csv
import torch
from cnsenti import Emotion, Sentiment
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re
import json

emotion = Emotion()
senti = Sentiment()

classifier = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-binary-chinese")

# Qwen
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)



def qwen_sentiment_standardized(text):
    """
    对 Qwen 的输出强制要求 JSON 格式并做稳健解析。
    返回 (full_text, answer, reason)：
      - full_text: 模型完整输出（用于调试）
      - answer: 标准化情感标签 '正向'/'负向'/'中性' 或 '未知'
      - reason: 一句简短的理由（若无法提取则空字符串）
    """
    # 严格化 prompt：要求 ONLY 输出 JSON（示例包含），有助于模型按固定格式返回
    prompt = f"""
你是一个情感分析助手。请**只**以合法的 JSON 对象输出结果（不要输出其他多余文字）。
JSON 格式必须包含两个字段：answer（取值仅限 "正向" / "负向" / "中性"）和 reason（一句话的简短理由）。
示例输出格式（必须严格遵守）：
{{"answer":"正向","reason":"文本包含明显的积极情绪词，如“开心”“高兴”。"}}

现在请对下面文本进行情感判断并按上面格式输出 JSON：

文本：{text}
"""
    # 编码并推理
    inputs = tokenizer(prompt, return_tensors="pt").to(qwen_model.device)
    outputs = qwen_model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 默认值
    answer = "未知"
    reason = ""

    # 1) 优先尝试从输出中提取 JSON 段（使用正则找第一个 { ... }）
    json_text = None
    try:
        # 找到最外层的大括号段（贪婪或非贪婪都试）
        m = re.search(r"\{(?:.|\s)*\}", full_text)
        if m:
            candidate = m.group(0)
            # 尝试修正常见中文引号或多余注释（很少见）
            candidate_fixed = candidate.replace('“', '"').replace('”', '"').replace("：", ":")
            # 解析 json
            parsed = json.loads(candidate_fixed)
            # 检查 key 存在性和合法性
            if isinstance(parsed, dict) and "answer" in parsed:
                a = parsed.get("answer", "")
                r = parsed.get("reason", "")
                print("获取成功" + r)
                if isinstance(a, str):
                    answer = a.strip()
                if isinstance(r, str):
                    reason = r.strip()
                    print("reason抓取成功" + reason)
                # 返回（若 answer 合法则认为成功）
                if answer in ("正向", "负向", "中性"):
                    return full_text, answer, reason
    except Exception:
        # 若 JSON 解析失败，继续走回退解析
        pass

    # 2) 回退解析：查找“答案：” 和 “理由： / 解析：”
    # 尝试提取最后出现的“答案：”后面的非空行作为 answer
    if "答案：" in full_text:
        try:
            # 取最后一个“答案：”之后的块，避免前面示例影响
            segment = full_text.rsplit("答案：", 1)[1]
            # 抽取第一行非空文本作为 answer（去掉句尾标点）
            for line in segment.splitlines():
                line = line.strip()
                if line:
                    # 可能模型写成“正向的”，只保留关键词
                    if "正" in line:
                        answer = "正向"
                    elif "负" in line:
                        answer = "负向"
                    elif "中" in line:
                        answer = "中性"
                    else:
                        # 直接把这一行作为 answer（兜底）
                        answer = line.strip()
                    break
        except Exception:
            pass

    # 理由解析：优先找“理由：”再找“解析：”，若仍为空则尝试抽取首个带引号或带破折号的句子
    if "理由：" in full_text:
        try:
            segment = full_text.split("理由：", 1)[1]
            # 取第一行非空作为理由
            for line in segment.splitlines():
                if line.strip():
                    reason = line.strip().strip("- ").strip('"“”')
                    break
        except Exception:
            pass
    elif "解析：" in full_text:
        try:
            segment = full_text.split("解析：", 1)[1]
            for line in segment.splitlines():
                if line.strip():
                    reason = line.strip().strip("- ").strip('"“”')
                    break
        except Exception:
            pass

    # 如果仍无 reason，尝试从 full_text 中抓取首条带引号或首个由“- ”开头的解释项
    if not reason:
        # 找 - "..." 或 - ... 的行
        for line in full_text.splitlines():
            s = line.strip()
            if s.startswith("-"):
                candidate = s.lstrip("- ").strip()
                if len(candidate) > 5:
                    reason = candidate
                    break
            # 带中文引号的直接取第一个引号内内容
            m2 = re.search(r'[“"](.*?)[”"]', s)
            if m2:
                candidate = m2.group(1).strip()
                if len(candidate) > 5:
                    reason = candidate
                    break

    # 最终清理：截断过长的理由为一小句（最多 200 字）
    if reason and len(reason) > 200:
        reason = reason[:200] + "..."

    # 如果 answer 还是未知, 试用更宽松的关键词映射
    if answer == "未知":
        low = full_text.lower()
        if "开心" in full_text or "高兴" in full_text or "正面" in full_text or "积极" in full_text:
            answer = "正向"
        elif "难过" in full_text or "悲" in full_text or "愤怒" in full_text or "负面" in full_text:
            answer = "负向"
        elif "中性" in full_text or "既不是" in full_text:
            answer = "中性"

    return full_text, answer, reason


def evaluate_three_models(text_list):
    results = []

    for text in text_list:
        print(f"Evaluating: {text}")

        # CNSenti emotion
        emo_res = emotion.emotion_count(text)

        # CNSenti pos/neg
        senti_res = senti.sentiment_count(text)

        # RoBERTa
        rob_res = classifier(text)[0]
        rob_label = rob_res["label"]
        rob_score = float(rob_res["score"])

        # Qwen
        qwen_full, qwen_ans, qwen_reason = qwen_sentiment_standardized(text)

        results.append({
            "text": text,
            "cnsenti_emotion": emo_res,
            "cnsenti_sentiment": senti_res,
            "roberta_label": rob_label,
            "roberta_score": rob_score,
            "qwen_answer": qwen_ans,
            "qwen_reason": qwen_reason,
            "qwen_full": qwen_full
        })

    return results

def save_to_csv(results, path="sentiment_eval.csv"):
    keys = results[0].keys()
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV saved to {path}")

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

results = evaluate_three_models(test_texts)
save_to_csv(results)