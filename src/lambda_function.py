#! /usr/bin/env python

import json, boto3


client = boto3.client("bedrock-runtime")

def handler(event, context):
    modelId = "anthropic.claude-3-haiku-20240307-v1:0"
    max_tokens = 20000
    payload = {
        "modelId": modelId,
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            # "text": "Write me a detailed description of these two photos, and then a poem talking about it."
                            "text": f"Write me a introducation about Beijing and Chongqing"
                        }
                    ]
                }
            ],
            "temperature": 0.5,
            "top_p": 0.999,
            "top_k": 250,
            "stop_sequences": ['\n\nHuman:']
        }
    }
    body_bytes = json.dumps(payload['body']).encode('utf-8')
    response = client.invoke_model_with_response_stream(
        body=body_bytes,
        modelId=payload['modelId'],
        accept="application/json",
        contentType="application/json",
    )
    status_code = response["ResponseMetadata"]["HTTPStatusCode"]
    # stream = response.get('body')

    if status_code != 200:
        raise ValueError(f"Error invoking Bedrock API: {status_code}")
    for response in response["body"]:
        json_response = json.loads(response["chunk"]["bytes"])

        print(json_response["generation"])
        yield json_response["generation"].encode()

