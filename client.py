# -*- coding: utf-8 -*-
# USAGE: python3 client.py

from openai import OpenAI


BASE_URL = "http://localhost:8621/v1"


def chat_completion(prompt, model=''):
    client = OpenAI(
        base_url=BASE_URL,
        api_key='EMPTY_KEY'
    )

    chat_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        top_p=0.9,
        max_tokens=512,
        extra_body={
            "repetition_penalty": 1.05,
        },
    )

    return chat_response


if __name__ == '__main__':
    response = chat_completion(prompt="抑郁症有哪些症状")
    content = response.choices[0].message.content
    print(content)
