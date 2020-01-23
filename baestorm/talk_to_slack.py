import requests


def notify_me(notification):
    headers = {"Content-type": "application/json"}

    data = {"text": notification}

    response = requests.post(
        "https://hooks.slack.com/services/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",
        headers=headers,
        data=str(data),
    )
    return response.content
